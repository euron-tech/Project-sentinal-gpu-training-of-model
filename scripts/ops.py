#!/usr/bin/env python3
"""SENTINEL one-tap operations (Windows, Linux, macOS, GitHub Actions). Needs: aws, terraform, docker (+ gh optional).

    python scripts/ops.py check              # read-only: settings, Hyperstack models, isolation in the region
    python scripts/ops.py up                 # everything: bootstrap -> AWS incl. Kinesis streams (Terraform) -> build +
                                             #   push images -> copy the labelled dataset to S3 -> print URLs
    python scripts/ops.py deploy [--tag T]   # build + push images, terraform apply, wait (what CI runs on every push)
    python scripts/ops.py dataset [--dir D]  # (re)copy the labelled SFT holdout split to S3 for the producer
    python scripts/ops.py status             # URLs, ECS service, health
    python scripts/ops.py down [--all] [-y]  # terraform destroy (only what this project created) [+ bootstrap stack]
    python scripts/ops.py audit              # list any AWS resource still tagged Project=<project>

Settings come from the environment or the git-ignored .env file in the repo root:
  TF_VAR_hyperstack_host / TF_VAR_model_api_key          Hyperstack VM IP + the key printed by serve_all.sh
  AWS_REGION (ap-south-1), PROJECT (sentinel), ENVIRONMENT (prod), GITHUB_REPO (owner/repo)
Any Terraform variable can be set as TF_VAR_<name>, e.g. TF_VAR_alert_email=me@example.com.
The models are not deployed to AWS: they run on the Hyperstack GPU VM (hyperstack/serve_all.sh).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INFRA, BOOT = ROOT / "infra", ROOT / "infra" / "bootstrap"
IMAGES = {"ui": ("ui", "ui/Dockerfile"), "stream": ("stream", "stream/Dockerfile")}  # repo -> (context, dockerfile)


def log(msg: str) -> None:
    print(f"\n\033[1;34m==> {msg}\033[0m", flush=True)


def run(cmd, cwd=None, capture=False, check=True, input_=None, quiet=False) -> str:
    if not quiet:
        print("$ " + " ".join(str(c) for c in cmd), flush=True)
    r = subprocess.run([str(c) for c in cmd], cwd=cwd or ROOT, text=True, input=input_,
                       stdout=subprocess.PIPE if capture else None, stderr=subprocess.PIPE if capture else None)
    if check and r.returncode != 0:
        if capture:
            sys.stderr.write((r.stderr or "") + (r.stdout or ""))
        sys.exit(f"command failed ({r.returncode}): {' '.join(str(c) for c in cmd[:4])} ...")
    return (r.stdout or "").strip() if capture else ""


def _load_env(path: Path):
    """KEY=VALUE lines from .env, without overriding variables that are already set."""
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            m = re.match(r"\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*?)\s*$", line)
            if m and not os.getenv(m.group(1)):
                os.environ[m.group(1)] = m.group(2).strip('"').strip("'")


class Ctx:
    def __init__(self, a):
        self.region, self.project, self.env = a.region, a.project, a.environment
        self.name = f"{self.project}-{self.env}"
        self.repo = a.github_repo
        os.environ["AWS_REGION"] = os.environ["AWS_DEFAULT_REGION"] = self.region
        os.environ["TF_VAR_region"], os.environ["TF_VAR_project"], os.environ["TF_VAR_environment"] = \
            self.region, self.project, self.env
        os.environ.setdefault("TF_IN_AUTOMATION", "1")
        self._account = None

    @property
    def account(self) -> str:
        if not self._account:
            self._account = run(["aws", "sts", "get-caller-identity", "--query", "Account", "--output", "text"],
                                capture=True, quiet=True)
        return self._account

    @property
    def registry(self) -> str:
        return f"{self.account}.dkr.ecr.{self.region}.amazonaws.com"

    @property
    def state_bucket(self) -> str:
        return os.getenv("TF_STATE_BUCKET") or f"{self.project}-tfstate-{self.account}-{self.region}"

    @property
    def data_bucket(self) -> str:
        return f"{self.name}-data-{self.account}"

    def ecr(self, repo: str) -> str:
        return f"{self.registry}/{self.name}/{repo}"


# ------------------------------------------------------------------ helpers
def preflight(need_docker=True, need_models=True):
    missing = [t for t in ["aws", "terraform"] + (["docker"] if need_docker else []) if not shutil.which(t)]
    if missing:
        sys.exit(f"missing tools on PATH: {', '.join(missing)}")
    if need_docker and subprocess.run(["docker", "info"], capture_output=True).returncode != 0:
        sys.exit("docker is not running")
    if need_models:
        if not os.getenv("TF_VAR_hyperstack_host"):
            print("note: TF_VAR_hyperstack_host is not set - the scorer has no models and holds every transaction "
                  "for REVIEW until you set it (and TF_VAR_model_api_key) and deploy again")


def check(c: Ctx) -> bool:
    """Verify every setting and that this project cannot collide with anything already in the account/region.
    Read-only: it creates nothing. Returns False if a blocking problem was found."""
    import urllib.request
    bad, warn = [], []
    good = lambda m: print(f"  \033[32mOK\033[0m   {m}")

    log(f"Checking configuration for {c.name} in {c.region}")
    for t in ["aws", "terraform", "docker"]:
        (good(f"{t} found") if shutil.which(t) else bad.append(f"{t} is not installed / not on PATH"))
    if shutil.which("docker") and subprocess.run(["docker", "info"], capture_output=True).returncode != 0:
        bad.append("Docker Desktop is not running")
    if shutil.which("gh") and subprocess.run(["gh", "auth", "status"], capture_output=True).returncode == 0:
        good("GitHub CLI logged in (CI/CD variables + secrets will be set)")
    else:
        warn.append("GitHub CLI not logged in (`gh auth login`): AWS deploys, but CI/CD is not wired to GitHub")

    ident = subprocess.run(["aws", "sts", "get-caller-identity", "--output", "json"], capture_output=True, text=True)
    if ident.returncode != 0:
        bad.append("AWS CLI has no valid credentials (`aws configure` or `aws login`)")
    else:
        good(f"AWS identity {json.loads(ident.stdout)['Arn']}  region {c.region}")


    host, key = os.getenv("TF_VAR_hyperstack_host", ""), os.getenv("TF_VAR_model_api_key", "")
    if not host or not key:
        warn.append("TF_VAR_hyperstack_host / TF_VAR_model_api_key missing: the dashboard deploys but every "
                    "transaction is held for REVIEW until the models are reachable")
    else:
        for name, port in [("LoRA", 8000), ("XGBoost", 8001), ("QLoRA", 8002)]:
            try:
                req = urllib.request.Request(f"http://{host}:{port}/health", headers={"X-API-Key": key})
                urllib.request.urlopen(req, timeout=6).read()
                good(f"Hyperstack {name} reachable on :{port}")
            except Exception as e:  # noqa: BLE001
                warn.append(f"Hyperstack {name} :{port} not reachable ({type(e).__name__}); start serve_all.sh and "
                            f"open TCP {port} on the Hyperstack firewall")

    if ident.returncode == 0:   # isolation checks (read-only)
        aws = lambda *a: subprocess.run(["aws", *a, "--region", c.region, "--output", "text"], capture_output=True, text=True)
        deployed = aws("s3api", "head-bucket", "--bucket", c.state_bucket).returncode == 0
        vpcs = len(aws("ec2", "describe-vpcs", "--query", "Vpcs[].VpcId").stdout.split())
        q = aws("service-quotas", "get-service-quota", "--service-code", "vpc", "--quota-code", "L-F678F1CE",
                "--query", "Quota.Value").stdout.strip()
        quota = int(float(q)) if q and q != "None" else 5
        if deployed:
            good(f"existing {c.name} state found: this is an update of the same stack")
        elif vpcs >= quota:
            bad.append(f"{vpcs}/{quota} VPCs already used in {c.region}: raise the VPC quota (this project needs its own)")
        else:
            good(f"VPC quota: {vpcs}/{quota} used, this project adds its own isolated VPC (10.40.0.0/16)")
        if not deployed:
            clashes = []
            if c.name in [a.rsplit("/", 1)[-1] for a in aws("ecs", "list-clusters", "--query", "clusterArns").stdout.split()]:
                clashes.append(f"ECS cluster {c.name}")
            clashes += [f"ECR {r}" for r in aws("ecr", "describe-repositories", "--query", "repositories[].repositoryName").stdout.split()
                        if r.startswith(c.name + "/")]
            clashes += [f"ALB {n}" for n in aws("elbv2", "describe-load-balancers", "--query", "LoadBalancers[].LoadBalancerName").stdout.split()
                        if n.startswith(c.name)]
            clashes += [f"IAM role {n}" for n in aws("iam", "list-roles", "--query", f"Roles[?starts_with(RoleName,'{c.name}')].RoleName").stdout.split()]
            clashes += [f"Kinesis stream {n}" for n in aws("kinesis", "list-streams", "--query", "StreamNames").stdout.split()
                        if n.startswith(c.name + "-")]
            clashes += [f"log group {n}" for n in aws("logs", "describe-log-groups", "--log-group-name-prefix", f"/ecs/{c.name}",
                                                      "--query", "logGroups[].logGroupName").stdout.split()]
            if clashes:
                bad.append("names this project would create already exist (not managed by this project): " + ", ".join(clashes))
            else:
                good(f"no existing resource uses the '{c.name}' names: nothing else in {c.region} will be touched")

    for w in warn:
        print(f"  \033[33mWARN\033[0m {w}")
    for b in bad:
        print(f"  \033[31mFAIL\033[0m {b}")
    print("\n  " + ("\033[32mready to deploy\033[0m" if not bad else "\033[31mfix the FAIL lines, then re-run\033[0m"))
    return not bad


def tf_version() -> tuple:
    v = json.loads(run(["terraform", "version", "-json"], capture=True, quiet=True))["terraform_version"]
    return tuple(int(x) for x in re.findall(r"\d+", v)[:2])


def tf_init(c: Ctx):
    args = ["terraform", f"-chdir={INFRA}", "init", "-input=false", "-reconfigure",
            f"-backend-config=bucket={c.state_bucket}", f"-backend-config=key={c.project}/{c.env}/terraform.tfstate",
            f"-backend-config=region={c.region}", "-backend-config=encrypt=true"]
    if tf_version() >= (1, 10):
        args.append("-backend-config=use_lockfile=true")    # S3-native state locking
    run(args)


def tf_outputs() -> dict:
    out = run(["terraform", f"-chdir={INFRA}", "output", "-json"], capture=True, check=False, quiet=True)
    try:
        return {k: v["value"] for k, v in json.loads(out or "{}").items()}
    except ValueError:
        return {}


def git_tag() -> str:
    if shutil.which("git"):
        sha = subprocess.run(["git", "rev-parse", "--short=10", "HEAD"], cwd=ROOT, capture_output=True, text=True)
        if sha.returncode == 0 and sha.stdout.strip():
            dirty = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
            return sha.stdout.strip() + ("-dirty" + time.strftime("%H%M%S") if dirty else "")
    return time.strftime("build-%Y%m%d-%H%M%S")


def detect_repo() -> str:
    if shutil.which("git"):
        url = subprocess.run(["git", "remote", "get-url", "origin"], cwd=ROOT, capture_output=True, text=True).stdout
        m = re.search(r"github\.com[:/]([^/]+/[^/.\s]+?)(?:\.git)?\s*$", url)
        if m:
            return m.group(1)
    return ""


def _owns_oidc() -> bool:
    """True if the GitHub OIDC provider is in the bootstrap state (created by us) -> keep managing it."""
    r = subprocess.run(["terraform", f"-chdir={BOOT}", "state", "list"], capture_output=True, text=True)
    return "aws_iam_openid_connect_provider.github" in (r.stdout or "")


def _s3_has(bucket: str, prefix: str) -> bool:
    r = subprocess.run(["aws", "s3api", "list-objects-v2", "--bucket", bucket, "--prefix", prefix, "--max-items", "1",
                        "--query", "KeyCount", "--output", "text"], capture_output=True, text=True)
    return r.returncode == 0 and r.stdout.strip() not in ("", "0", "None")


# ------------------------------------------------------------------ commands
def bootstrap(c: Ctx):
    log("Bootstrap: Terraform state bucket + GitHub Actions OIDC role")
    providers = run(["aws", "iam", "list-open-id-connect-providers", "--output", "json"], capture=True, quiet=True)
    has_oidc = "token.actions.githubusercontent.com" in providers
    run(["terraform", f"-chdir={BOOT}", "init", "-input=false"])
    # create the account-wide GitHub OIDC provider only if nobody has one; never take over someone else's
    create = _owns_oidc() or not has_oidc
    run(["terraform", f"-chdir={BOOT}", "apply", "-auto-approve", "-input=false",
         f"-var=github_repo={c.repo}", f"-var=create_oidc_provider={'true' if create else 'false'}"])
    out = {k: v["value"] for k, v in json.loads(run(["terraform", f"-chdir={BOOT}", "output", "-json"],
                                                    capture=True, quiet=True)).items()}
    if c.repo and out.get("github_role_arn") and shutil.which("gh"):
        log(f"Configuring GitHub repo {c.repo} (Actions variables)")
        for k, v in {"AWS_ROLE_ARN": out["github_role_arn"], "AWS_REGION": c.region, "PROJECT": c.project,
                     "ENVIRONMENT": c.env, "HYPERSTACK_HOST": os.getenv("TF_VAR_hyperstack_host", "")}.items():
            if v:
                run(["gh", "variable", "set", k, "--body", v, "-R", c.repo], check=False)
        for k, env in {"MODEL_API_KEY": "TF_VAR_model_api_key"}.items():
            if os.getenv(env):     # value goes in via stdin, never on the command line
                run(["gh", "secret", "set", k, "-R", c.repo], input_=os.environ[env], check=False)
    return out


def build_push(c: Ctx, tag: str, only=None):
    log(f"Build + push images to ECR ({c.registry}) tag {tag}")
    pw = run(["aws", "ecr", "get-login-password", "--region", c.region], capture=True, quiet=True)
    run(["docker", "login", "--username", "AWS", "--password-stdin", c.registry], input_=pw)
    for repo, (ctx, dockerfile) in IMAGES.items():
        if only and repo not in only:
            continue
        url = c.ecr(repo)
        run(["docker", "build", "--platform", "linux/amd64", "-f", dockerfile, "-t", f"{url}:{tag}", "-t", f"{url}:latest", ctx])
        run(["docker", "push", f"{url}:{tag}"])
        run(["docker", "push", f"{url}:latest"])


def seed_dataset(c: Ctx, data_dir: Path = ROOT / "data" / "sft", force=False):
    """Copy the labelled holdout split to S3 once (local data/sft, or download it from Supabase with .env creds)."""
    if not force and _s3_has(c.data_bucket, "data/sft/holdout/"):
        return
    if not any((data_dir / "holdout").glob("*.jsonl.gz")):
        if not (os.getenv("SUPABASE_URL") and os.getenv("SUPABASE_KEY")):
            print("  dataset: none in data/sft and no SUPABASE_URL/KEY -> the producer streams synthetic traffic only")
            return
        log("Downloading the labelled holdout split from Supabase")
        sys.path.insert(0, str(ROOT / "xgboost_benchmark"))
        from data import download_split  # noqa: E402
        download_split("holdout", data_dir)
    log(f"Uploading dataset to s3://{c.data_bucket}/data/sft/holdout/")
    run(["aws", "s3", "sync", data_dir / "holdout", f"s3://{c.data_bucket}/data/sft/holdout/", "--only-show-errors"])


def deploy(c: Ctx, tag: str, wait=True, only=None):
    preflight()
    tf_init(c)
    log("Create registries + dataset bucket first (images must exist before the service)")
    run(["terraform", f"-chdir={INFRA}", "apply", "-auto-approve", "-input=false",
         "-target=aws_ecr_repository.this", "-target=aws_s3_bucket.data"])
    build_push(c, tag, only)
    seed_dataset(c)
    log("terraform apply (Kinesis streams, VPC, ALB, ECS Fargate, CloudWatch, SNS)")
    run(["terraform", f"-chdir={INFRA}", "apply", "-auto-approve", "-input=false", f"-var=image_tag={tag}"])
    out = tf_outputs()
    if wait and out.get("services"):
        log("Waiting for the ECS Fargate service to become stable (~3-5 min)")
        if subprocess.run(["aws", "ecs", "wait", "services-stable", "--cluster", c.name,
                           "--services", *out["services"]]).returncode != 0:
            print("  (not stable yet - check `python scripts/ops.py status`)")
    summary(out)
    return out


def summary(out: dict):
    log("SENTINEL is deployed")
    for k in ("dashboard_url", "flow_url", "analytics_api", "cloudwatch_dashboard", "kinesis_streams", "model_urls"):
        if out.get(k):
            print(f"  {k:22s} {out[k]}")
    print("  Open the dashboard -> Live Flow -> 'Start auto traffic' to send transactions through Kinesis.")


def dataset(c: Ctx, data_dir=None):
    preflight(need_docker=False, need_models=False)
    seed_dataset(c, Path(data_dir) if data_dir else ROOT / "data" / "sft", force=True)
    run(["aws", "ecs", "update-service", "--cluster", c.name, "--service", f"{c.name}-app", "--force-new-deployment",
         "--query", "service.serviceName", "--output", "text"], check=False)
    print("Service restarted: the producer now mixes the dataset into the traffic.")


def status(c: Ctx):
    preflight(need_docker=False, need_models=False)
    tf_init(c)
    out = tf_outputs()
    if not out:
        print("Not deployed (no Terraform outputs).")
        return
    summary(out)
    run(["aws", "ecs", "describe-services", "--cluster", c.name, "--services", *out["services"],
         "--query", "services[].{service:serviceName,status:status,running:runningCount,desired:desiredCount,"
                    "rollout:deployments[0].rolloutState}", "--output", "table"], check=False, quiet=True)


def down(c: Ctx, all_=False, yes=False):
    preflight(need_docker=False)
    if not yes:
        ans = input(f"Destroy every resource of '{c.name}' in {c.region} (account {c.account}), "
                    f"including its Kinesis streams? Type the name to confirm: ")
        if ans.strip() != c.name:
            sys.exit("aborted")
    if subprocess.run(["aws", "s3api", "head-bucket", "--bucket", c.state_bucket], capture_output=True).returncode == 0:
        tf_init(c)
        log(f"terraform destroy {c.name} (only resources in this project's state)")
        run(["terraform", f"-chdir={INFRA}", "destroy", "-auto-approve", "-input=false"])
    else:
        print(f"state bucket {c.state_bucket} not found - main stack already gone")
    if all_:
        log("Destroying the bootstrap stack (state bucket + CI role)")
        if (BOOT / "terraform.tfstate").exists():
            run(["terraform", f"-chdir={BOOT}", "init", "-input=false"])
            run(["terraform", f"-chdir={BOOT}", "destroy", "-auto-approve", "-input=false", f"-var=github_repo={c.repo}",
                 f"-var=create_oidc_provider={'true' if _owns_oidc() else 'false'}"])
        else:
            print("  no local bootstrap state (run `down --all` on the machine that ran `up`)")
    audit(c)


def audit(c: Ctx):
    log(f"AWS resources still tagged Project={c.project} in {c.region}")
    r = run(["aws", "resourcegroupstaggingapi", "get-resources", "--tag-filters", f"Key=Project,Values={c.project}",
             "--query", "ResourceTagMappingList[].ResourceARN", "--output", "text"], capture=True, check=False, quiet=True)
    arns = [a for a in r.split() if a]
    print("\n".join("  " + a for a in arns) if arns else "  none")
    print("  (the tagging index can lag a few minutes after a destroy)")


def main(argv=None):
    _load_env(ROOT / ".env")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--region", default=os.getenv("AWS_REGION", "ap-south-1"))
    p.add_argument("--project", default=os.getenv("PROJECT", "sentinel"))
    p.add_argument("--environment", default=os.getenv("ENVIRONMENT", "prod"))
    p.add_argument("--github-repo", default=os.getenv("GITHUB_REPO") or detect_repo(),
                   help="owner/repo allowed to deploy from GitHub Actions")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("up"); s.add_argument("--tag"); s.add_argument("--no-wait", action="store_true")
    sub.add_parser("check", help="read-only: verify settings, models and isolation before deploying")
    sub.add_parser("bootstrap")
    s = sub.add_parser("deploy"); s.add_argument("--tag"); s.add_argument("--no-wait", action="store_true")
    s.add_argument("--only", help="comma list of images to rebuild (ui,stream)")
    s = sub.add_parser("dataset"); s.add_argument("--dir", help="folder with holdout/*.jsonl.gz (default data/sft)")
    sub.add_parser("status")
    s = sub.add_parser("down"); s.add_argument("--all", action="store_true"); s.add_argument("-y", "--yes", action="store_true")
    sub.add_parser("audit")
    a = p.parse_args(argv)
    c = Ctx(a)
    if a.cmd == "check":
        sys.exit(0 if check(c) else 1)
    if a.cmd == "up":
        if not check(c):
            sys.exit("pre-deploy check failed - nothing was created")
        preflight()
        bootstrap(c)
        deploy(c, a.tag or git_tag(), wait=not a.no_wait)
    elif a.cmd == "bootstrap":
        preflight(need_docker=False, need_models=False)
        bootstrap(c)
    elif a.cmd == "deploy":
        deploy(c, a.tag or git_tag(), wait=not a.no_wait, only=a.only.split(",") if a.only else None)
    elif a.cmd == "dataset":
        dataset(c, a.dir)
    elif a.cmd == "status":
        status(c)
    elif a.cmd == "down":
        down(c, a.all, a.yes)
    elif a.cmd == "audit":
        audit(c)


if __name__ == "__main__":
    main()
