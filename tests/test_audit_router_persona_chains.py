from __future__ import annotations

import json

from tools.audit_router_persona_chains import audit_manifest, build_summary, load_completed_records


def test_audit_manifest_distinguishes_chain_statuses(tmp_path) -> None:
    manifest = [
        {'assignment_id': 'c1::round01', 'chain_id': 'c1', 'evolution_round': 1},
        {'assignment_id': 'c1::round02', 'chain_id': 'c1', 'evolution_round': 2},
        {'assignment_id': 'c1::round03', 'chain_id': 'c1', 'evolution_round': 3},
        {'assignment_id': 'c2::round01', 'chain_id': 'c2', 'evolution_round': 1},
        {'assignment_id': 'c2::round02', 'chain_id': 'c2', 'evolution_round': 2},
        {'assignment_id': 'c2::round03', 'chain_id': 'c2', 'evolution_round': 3},
        {'assignment_id': 'c3::round01', 'chain_id': 'c3', 'evolution_round': 1},
        {'assignment_id': 'c3::round02', 'chain_id': 'c3', 'evolution_round': 2},
        {'assignment_id': 'c3::round03', 'chain_id': 'c3', 'evolution_round': 3},
        {'assignment_id': 'c4::round01', 'chain_id': 'c4', 'evolution_round': 1},
        {'assignment_id': 'c4::round02', 'chain_id': 'c4', 'evolution_round': 2},
        {'assignment_id': 'c4::round03', 'chain_id': 'c4', 'evolution_round': 3},
    ]

    root_a = tmp_path / 'root_a'
    root_b = tmp_path / 'root_b'
    (root_a / 'job1').mkdir(parents=True)
    (root_b / 'job2').mkdir(parents=True)

    (root_a / 'job1' / 'results.jsonl').write_text(
        '\n'.join(
            [
                json.dumps({'assignment_id': 'c1::round01', 'chain_id': 'c1'}),
                json.dumps({'assignment_id': 'c1::round02', 'chain_id': 'c1'}),
                json.dumps({'assignment_id': 'c1::round03', 'chain_id': 'c1'}),
                json.dumps({'assignment_id': 'c2::round01', 'chain_id': 'c2'}),
                json.dumps({'assignment_id': 'c3::round01', 'chain_id': 'c3'}),
                json.dumps({'assignment_id': 'c3::round02', 'chain_id': 'c3'}),
            ]
        ),
        encoding='utf-8',
    )
    (root_b / 'job2' / 'results.jsonl').write_text(
        '\n'.join(
            [
                json.dumps({'assignment_id': 'c2::round02', 'chain_id': 'c2'}),
                json.dumps({'assignment_id': 'c2::round03', 'chain_id': 'c2'}),
            ]
        ),
        encoding='utf-8',
    )

    completed = load_completed_records([str(root_a), str(root_b)])
    chain_rows, rerun_manifest = audit_manifest(
        manifest,
        completed,
        rerun_statuses={'partial_compliant', 'noncompliant', 'missing'},
    )
    by_chain = {row['chain_id']: row for row in chain_rows}
    summary = build_summary(chain_rows, rerun_manifest)

    assert by_chain['c1']['status'] == 'complete_compliant'
    assert by_chain['c2']['status'] == 'noncompliant'
    assert by_chain['c3']['status'] == 'partial_compliant'
    assert by_chain['c4']['status'] == 'missing'

    rerun_ids = [row['assignment_id'] for row in rerun_manifest]
    assert rerun_ids == [
        'c2::round01',
        'c2::round02',
        'c2::round03',
        'c3::round01',
        'c3::round02',
        'c3::round03',
        'c4::round01',
        'c4::round02',
        'c4::round03',
    ]
    assert summary == {
        'chain_count': 4,
        'assignment_count': 12,
        'complete_compliant_chain_count': 1,
        'partial_compliant_chain_count': 1,
        'noncompliant_chain_count': 1,
        'missing_chain_count': 1,
        'complete_compliant_assignment_count': 3,
        'partial_compliant_completed_assignment_count': 2,
        'noncompliant_assignment_count': 3,
        'rerun_chain_count': 3,
        'rerun_assignment_count': 9,
    }
