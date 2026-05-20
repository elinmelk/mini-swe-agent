"""Docker-free mini-benchmark.

Each fixture lives in ``fixtures/<instance_id>/`` with:

  instance.yaml      — { instance_id, problem_statement, test_command, ... }
  repo/              — the buggy repo tree (copied into a tmpdir per run)

The runner:
  1. Copies repo/ to a tmpdir, ``git init && git add . && git commit``.
  2. Runs the agent in that tmpdir using LocalEnvironment.
  3. Captures the agent's final patch (from the submission).
  4. Re-applies the patch to a clean copy, runs ``test_command``,
     and records "resolved" iff the test command exited 0 *and* the same test
     command failed before the patch.

This keeps the protocol close to SWE-Bench-Lite (a patch is the deliverable,
resolution is verified by tests) but needs nothing beyond Python and pytest.
"""
