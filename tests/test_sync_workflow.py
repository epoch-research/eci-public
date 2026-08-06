from pathlib import Path


WORKFLOW = Path(__file__).parents[1] / ".github" / "workflows" / "sync-to-public.yml"


def test_sync_uses_the_proven_public_repository_token() -> None:
    workflow = WORKFLOW.read_text()

    assert "PUBLIC_REPO_TOKEN: ${{ secrets.PUBLIC_REPO_TOKEN }}" in workflow
    assert "actions/create-github-app-token@" not in workflow
    assert "EPOCH_AUTOMATIONS_WRITER_PRIVATE_KEY" not in workflow
    assert "EPOCH_AUTOMATIONS_WRITER_TOKEN" not in workflow


def test_sync_keeps_credentials_out_of_checkout_and_remote_url() -> None:
    workflow = WORKFLOW.read_text()
    sync_job = workflow.split("\n  sync:\n", maxsplit=1)[1]

    assert "    permissions:\n      contents: read\n" in sync_job
    assert "persist-credentials: false" in workflow
    assert "https://github.com/epoch-research/eci-public.git" in workflow
    assert (
        'url."https://x-access-token:${PUBLIC_REPO_TOKEN}@github.com/".insteadOf '
        '"https://github.com/"'
    ) in workflow
    assert "git push public main:main" in workflow
