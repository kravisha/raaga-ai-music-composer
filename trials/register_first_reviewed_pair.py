"""Register the approved derivative without treating the original as extra data."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

DATA = Path('C:/Users/Krish/ClaudeStuff/Raaga-SPB-Direct-TTS-20260910')
path = DATA/'data/manifest.json'
backup = DATA/'data/manifest-before-first-approved-pair.json'
pair = json.loads((DATA/'data/paired-speech-01/pairing.json').read_text(encoding='utf-8'))
for field, hash_field in [('path','sha256'), ('source_path','source_sha256'),
                         ('text_file','text_sha256'), ('parent_clip_path','parent_clip_sha256'),
                         ('approved_full_verbatim_file','approved_full_verbatim_sha256')]:
    assert hashlib.sha256(Path(pair[field]).read_bytes()).hexdigest() == pair[hash_field]
manifest = json.loads(path.read_text(encoding='utf-8'))
assert not any(c['id']==pair['id'] for c in manifest['clips'])
assert not backup.exists()
backup.write_bytes(path.read_bytes())
parent = next(c for c in manifest['clips'] if c['id']==pair['parent_clip_id'])
parent.update(text_review='human_approved_working_text',
              approval_file='data/review-batch-01/cc-sbs2-060000-approval.json',
              paired_derivative_id=pair['id'],
              quality_review='reviewed; incomplete tail excluded in paired derivative',
              speaker_review='reviewed', target_speaker='SPB',
              accepted_for_training=False, split='review_only')
manifest['clips'].append(pair)
manifest.update(updated_utc=datetime.now(timezone.utc).isoformat(),
                status='first_training_pair_approved; evaluation_transcript_pending; validator_under_review',
                training_ready=False, training_started=False,
                admitted_training_seconds=pair['duration_seconds'],
                admitted_evaluation_seconds=0,
                draft_seconds=60.0, unique_review_candidate_source_seconds=80.0,
                excluded_first_clip_tail_source_seconds=80-pair['end_seconds'],
                duration_accounting='Original first20s review clip and its18.686s derivative overlap; never add both. Three other20s windows remain pending. Listening context copies are excluded.',
                readiness_blockers=['No approved evaluation text/audio pair yet',
                                    'Independent export-validator review has four remaining defects',
                                    'Bounded real-data pilot has not run'])
path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps({k:manifest[k] for k in ['status','training_ready','admitted_training_seconds','draft_seconds']}))
