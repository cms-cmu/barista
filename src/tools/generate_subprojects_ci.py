#!/usr/bin/env python3
"""Generate dynamic child pipeline YAML for subprojects (coffea4bees & bbreww).

Resolves the target branch for each subproject:
- Uses the current branch ($CI_MERGE_REQUEST_SOURCE_BRANCH_NAME or $CI_COMMIT_REF_NAME)
  if it exists on the remote repository.
- Otherwise falls back to 'master'.

Embeds self-contained base jobs (voms_proxy, setup_coffea4bees, setup_bbreww, .base_analysis_*)
so the child pipeline has zero dependency on local files and always succeeds in parsing.
"""
import os
import subprocess
import sys


def resolve_branch(repo_name, target_branch):
    """Check if target_branch exists in gitlab.cern.ch/cms-cmu/<repo_name>.

    Returns target_branch if found, else 'master'.
    """
    token = os.environ.get("CI_JOB_TOKEN")
    if token:
        remote_url = f"https://gitlab-ci-token:{token}@gitlab.cern.ch/cms-cmu/{repo_name}.git"
    else:
        user = os.environ.get(f"DEPLOY_TOKEN_USERNAME_{repo_name.upper()}")
        pwd = os.environ.get(f"DEPLOY_TOKEN_PASSWORD_{repo_name.upper()}")
        if user and pwd:
            remote_url = f"https://{user}:{pwd}@gitlab.cern.ch/cms-cmu/{repo_name}.git"
        else:
            remote_url = f"ssh://git@gitlab.cern.ch:7999/cms-cmu/{repo_name}.git"

    cmd = ["git", "ls-remote", "--exit-code", "--heads", remote_url, target_branch]
    try:
        res = subprocess.run(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=30,
        )
        if res.returncode == 0:
            print(f"[{repo_name}] Branch '{target_branch}' exists. Using ref: '{target_branch}'")
            return target_branch
    except Exception as e:
        print(f"[{repo_name}] Warning: git ls-remote check encountered an issue: {e}")

    print(f"[{repo_name}] Branch '{target_branch}' not found. Falling back to ref: 'master'")
    return "master"


BASE_DEFINITIONS = """
voms_proxy:
  stage: setup
  image: gitlab-registry.cern.ch/cms-cloud/cmssw-docker/al9-cms:latest
  tags:
    - k8s-cvmfs
  artifacts:
    paths:
      - proxy
  script:
    - mkdir -p ${HOME}/.globus
    - printf "${GRID_USERCERT}" | base64 -d > ${HOME}/.globus/usercert.pem
    - printf "${GRID_USERKEY}" | base64 -d > ${HOME}/.globus/userkey.pem
    - chmod 400 ${HOME}/.globus/userkey.pem
    - printf "${GRID_PASSWORD}" | base64 -d | voms-proxy-init --voms cms --pwstdin
    - voms-proxy-info --all
    - export VOMSPROXY=$(voms-proxy-info -path)
    - mkdir proxy
    - cp ${VOMSPROXY} proxy/x509_proxy

setup_coffea4bees:
  stage: setup
  image: gitlab-registry.cern.ch/cms-cmu/barista:${ANALYSIS_IMAGE_TAG}
  needs:
    - voms_proxy
  script:
    - COFFEA4BEES_DEFAULT_BRANCH="master"
    - echo "Checking for branch '$COFFEA4BEES_BRANCH' in coffea4bees..."
    - git ls-remote --exit-code --heads https://gitlab-ci-token:${CI_JOB_TOKEN}@gitlab.cern.ch/cms-cmu/coffea4bees.git "$COFFEA4BEES_BRANCH" || COFFEA4BEES_BRANCH="$COFFEA4BEES_DEFAULT_BRANCH"
    - echo "Cloning coffea4bees from branch '$COFFEA4BEES_BRANCH'"
    - git clone --depth=1 --branch $COFFEA4BEES_BRANCH https://${DEPLOY_TOKEN_USERNAME_COFFEA4BEES}:${DEPLOY_TOKEN_PASSWORD_COFFEA4BEES}@gitlab.cern.ch/cms-cmu/coffea4bees.git coffea4bees/
    - ls coffea4bees/
    # Bootstrap Pixi environment
    - echo "Installing Pixi and dependencies"
    - export PIXI_NO_PATH_UPDATE=1
    - curl -fsSL https://pixi.sh/install.sh | PIXI_HOME=$CI_PROJECT_DIR/.pixi-bin bash
    - export PATH="$CI_PROJECT_DIR/.pixi-bin/bin:$PATH"
    - pixi install
  artifacts:
    paths:
      - coffea4bees/
      - proxy/
      - .pixi/
      - .pixi-bin/
    expire_in: 4 hours

setup_bbreww:
  stage: setup
  image: gitlab-registry.cern.ch/cms-cmu/barista:${ANALYSIS_IMAGE_TAG}
  needs:
    - voms_proxy
  script:
    - BBREWW_DEFAULT_BRANCH="master"
    - echo "Checking for branch '$BBREWW_BRANCH' in bbreww..."
    - git ls-remote --exit-code --heads https://gitlab-ci-token:${CI_JOB_TOKEN}@gitlab.cern.ch/cms-cmu/bbreww.git "$BBREWW_BRANCH" || BBREWW_BRANCH="$BBREWW_DEFAULT_BRANCH"
    - echo "Cloning bbreww from branch '$BBREWW_BRANCH'"
    - git clone --depth=1 --branch $BBREWW_BRANCH https://${DEPLOY_TOKEN_USERNAME_BBREWW}:${DEPLOY_TOKEN_PASSWORD_BBREWW}@gitlab.cern.ch/cms-cmu/bbreww.git bbreww/
    - ls bbreww/
    # Bootstrap Pixi environment
    - echo "Installing Pixi and dependencies"
    - export PIXI_NO_PATH_UPDATE=1
    - curl -fsSL https://pixi.sh/install.sh | PIXI_HOME=$CI_PROJECT_DIR/.pixi-bin bash
    - export PATH="$CI_PROJECT_DIR/.pixi-bin/bin:$PATH"
    - pixi install
  artifacts:
    paths:
      - bbreww/
      - proxy/
      - .pixi/
      - .pixi-bin/
    expire_in: 4 hours

.base_analysis_coffea4bees:
  needs:
    - setup_coffea4bees
    - job: build_container
      optional: true
      artifacts: true
  image: ${CI_REGISTRY_IMAGE}:${ANALYSIS_IMAGE_TAG}
  variables:
    CLASSIFIER_WFS: "coffea4bees/classifier/test/config/workflow/CI"
    CLASSIFIER_CONFIG_PATHS: "coffea4bees"
  before_script:
    - echo "Add error handling flags"
    - set -e
    - set -o pipefail
    - export CONTAINER_PYTHON=$(which python)
    - export PATH="$PWD/.pixi-bin/bin:$PATH"
    - echo "Checking output directory"
    - ls $CI_PROJECT_PATH/output/ || true
    - echo "Setting analysis framework"
    - pwd && ls -la
  tags:
    - k8s-cvmfs
  retry: 2

.base_analysis_bbreww:
  needs:
    - setup_bbreww
    - job: build_container
      optional: true
      artifacts: true
  image: ${CI_REGISTRY_IMAGE}:${ANALYSIS_IMAGE_TAG}
  variables:
    CLASSIFIER_WFS: "bbreww/classifier/test/config/workflow/CI"
    CLASSIFIER_CONFIG_PATHS: "bbreww"
  before_script:
    - echo "Add error handling flags"
    - set -e
    - set -o pipefail
    - export CONTAINER_PYTHON=$(which python)
    - export PATH="$PWD/.pixi-bin/bin:$PATH"
    - echo "Checking output directory"
    - ls $CI_PROJECT_PATH/output/ || true
    - echo "Setting analysis framework"
    - pwd && ls -la
  tags:
    - k8s-cvmfs
  retry: 2

.base_combine_coffea4bees:
  image: gitlab-registry.cern.ch/cms-analysis/general/combine-container:CMSSW_11_3_4-combine_v9.1.0-harvester_v2.1.0
  needs:
    - setup_coffea4bees
  before_script:
    - echo "Checking output directory"
    - ls $CI_PROJECT_DIR/output/ || true
    - echo "Setting combine framework"
    - source /cvmfs/cms.cern.ch/cmsset_default.sh
    - cd /home/cmsusr/CMSSW_11_3_4/
    - cmsenv || true
    - cd $CI_PROJECT_DIR/
    - pwd && ls -la
    - ls -la coffea4bees/
  tags:
    - k8s-cvmfs
  retry: 2
"""


def main():
    target_branch = (
        os.environ.get("CI_MERGE_REQUEST_SOURCE_BRANCH_NAME")
        or os.environ.get("CI_COMMIT_REF_NAME")
        or "master"
    )
    print(f"Resolving subproject branches for target branch: '{target_branch}'")

    coffea4bees_ref = resolve_branch("coffea4bees", target_branch)
    bbreww_ref = resolve_branch("bbreww", target_branch)

    output_path = sys.argv[1] if len(sys.argv) > 1 else ".subprojects-ci.yml"

    content = f"""# Dynamic Child Pipeline for Subprojects
# Auto-generated by src/tools/generate_subprojects_ci.py
# Target branch:   {target_branch}
# coffea4bees ref: {coffea4bees_ref}
# bbreww ref:      {bbreww_ref}

variables:
  CONTAINER_HOST: cms-cmu
  RECREATE_CONTAINERS: "false"
  DEFAULT_ANALYSIS_IMAGE_TAG: "latest"
  ANALYSIS_IMAGE_TAG: ${{DEFAULT_ANALYSIS_IMAGE_TAG}}
  GIT_SUBMODULE_STRATEGY: recursive
  GIT_SUBMODULE_DEPTH: 1
  GIT_SUBMODULE_FORCE_HTTPS: "true"
  COFFEA4BEES_BRANCH: '{coffea4bees_ref}'
  BBREWW_BRANCH: '{bbreww_ref}'
  DATASET: coffea4bees/metadata/datasets/
  CLASSIFIER_WFS: coffea4bees/classifier/test/config/workflow/CI/
  CLASSIFIER_CONFIG_PATHS: coffea4bees

stages:
  - setup
  - code
  - classifier
  - skimmer
  - friendtree
  - analysis
  - tools
  - mixeddata
  - plot
  - cutflow
  - workflows

include:
  - project: 'cms-cmu/coffea4bees'
    ref: '{coffea4bees_ref}'
    file: 'workflows/gitlab-CI/stages_for_barista.yml'
  - project: 'cms-cmu/bbreww'
    ref: '{bbreww_ref}'
    file: 'workflows/gitlab-CI/stages_for_barista.yml'

{BASE_DEFINITIONS}
"""

    with open(output_path, "w") as f:
        f.write(content)

    print(f"Generated {output_path} successfully.")


if __name__ == "__main__":
    main()
