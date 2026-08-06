from pathlib import Path


WORKFLOW = Path(__file__).parents[1] / ".github" / "workflows" / "sync-to-public.yml"
APP_TOKEN_ACTION = (
    "actions/create-github-app-token@"
    "bcd2ba49218906704ab6c1aa796996da409d3eb1"
)


def test_sync_uses_repository_scoped_automations_writer_token() -> None:
    workflow = WORKFLOW.read_text()

    assert APP_TOKEN_ACTION in workflow
    assert "app-id: 4497189" in workflow
    assert (
        "private-key: ${{ secrets.EPOCH_AUTOMATIONS_WRITER_PRIVATE_KEY }}"
        in workflow
    )
    assert "owner: epoch-research" in workflow
    assert "repositories: eci-public" in workflow
    assert "permission-contents: write" in workflow
    assert "${{ secrets.PUBLIC_REPO_TOKEN }}" not in workflow


def test_sync_keeps_credentials_out_of_checkout_and_remote_url() -> None:
    workflow = WORKFLOW.read_text()
    sync_job = workflow.split("\n  sync:\n", maxsplit=1)[1]

    assert "    permissions:\n      contents: read\n" in sync_job
    assert "persist-credentials: false" in workflow
    assert "https://github.com/epoch-research/eci-public.git" in workflow
    assert "x-access-token:%s" in workflow
    assert '"$EPOCH_AUTOMATIONS_WRITER_TOKEN"' in workflow
    assert 'git -c http.https://github.com/.extraheader="$auth_header" push' in workflow
    assert 'url."https://x-access-token:' not in workflow
