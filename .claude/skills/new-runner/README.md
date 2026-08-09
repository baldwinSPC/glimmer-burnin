This skill is a thin wrapper over `docs/dev/new-testkind-playbook.md`.

It deliberately holds no rules of its own: two copies of a rule become two
different rules, and the guards in `runners/pins_test.go` track the document, not
this file. `hack/playbook/playbook_test.go` fails if this skill starts restating
the playbook instead of pointing at it.
