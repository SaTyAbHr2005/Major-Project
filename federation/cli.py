import argparse
import sys
from federation.storage import FederationStorage
from federation.jobs import JobManager
from federation.rounds import RoundManager
from federation.intake import IntakeManager
from federation.validation import UpdateValidator
from federation.security_hooks import M12SecurityHook
from federation.events import EventEmitter
from federation.aggregation import Aggregator
from federation.evaluation import EvaluationManager
from federation.promotion import PromotionManager, PromotionPolicy

def main():
    parser = argparse.ArgumentParser(description="Secure Federation Layer CLI")
    subparsers = parser.add_subparsers(dest="command")

    # job
    create_job_parser = subparsers.add_parser("create-job")
    create_job_parser.add_argument("--job-id", required=True)
    create_job_parser.add_argument("--task", required=True)
    create_job_parser.add_argument("--architecture", required=True)
    create_job_parser.add_argument("--num-classes", type=int, required=True)
    create_job_parser.add_argument("--min-participants", type=int, default=2)
    create_job_parser.add_argument("--expected-participants", type=str, default="", help="Comma-separated list of hospital IDs")

    # round
    create_round_parser = subparsers.add_parser("create-round")
    create_round_parser.add_argument("--job-id", required=True)
    create_round_parser.add_argument("--round-id", required=True)
    create_round_parser.add_argument("--round-number", type=int, required=True)
    create_round_parser.add_argument("--min-participants", type=int, default=2)
    create_round_parser.add_argument("--expected-participants", type=str, default="", help="Comma-separated list of hospital IDs")
    
    # submit handoff
    submit_parser = subparsers.add_parser("submit-update")
    submit_parser.add_argument("--job-id", required=True)
    submit_parser.add_argument("--round-id", required=True)
    submit_parser.add_argument("--participant-id", required=True)
    submit_parser.add_argument("--handoff-dir", required=True)
    
    # aggregate
    agg_parser = subparsers.add_parser("aggregate")
    agg_parser.add_argument("--job-id", required=True)
    agg_parser.add_argument("--round-id", required=True)

    # eval
    eval_parser = subparsers.add_parser("evaluate")
    eval_parser.add_argument("--global-model-id", required=True)
    
    # promote
    promote_parser = subparsers.add_parser("promote")
    promote_parser.add_argument("--global-model-id", required=True)
    promote_parser.add_argument("--eval-record-id", required=True)

    args = parser.parse_args()
    
    storage = FederationStorage()
    events = EventEmitter()
    
    if args.command == "create-job":
        jm = JobManager(storage)
        participants = [p.strip() for p in args.expected_participants.split(",")] if args.expected_participants else []
        jm.create_job(args.job_id, args.task, "image_classification", args.architecture, args.num_classes, {}, participants, args.min_participants, "", {}, {})
        print(f"Created job {args.job_id}")
        
    elif args.command == "create-round":
        rm = RoundManager(storage)
        participants = [p.strip() for p in args.expected_participants.split(",")] if args.expected_participants else []
        rm.create_round(args.job_id, args.round_id, args.round_number, participants, args.min_participants, "")
        rm.open_round(args.round_id)
        print(f"Created and opened round {args.round_id}")
        
    elif args.command == "submit-update":
        intake = IntakeManager(storage, UpdateValidator(storage), M12SecurityHook(storage=storage), events)
        update = intake.receive_handoff(args.job_id, args.round_id, args.participant_id, args.handoff_dir)
        print(f"Submitted update: {update.update_id} - status: {update.status}")
        
    elif args.command == "aggregate":
        rm = RoundManager(storage)
        rm.check_deadline_and_participants(args.round_id)
        rm.start_aggregation(args.round_id)
        agg = Aggregator(storage, events)
        gm = agg.aggregate_round(args.job_id, args.round_id)
        print(f"Aggregated! Global Model ID: {gm.global_model_id}")
        
    elif args.command == "evaluate":
        ev = EvaluationManager(storage, events)
        record = ev.evaluate(args.global_model_id)
        print(f"Evaluated! Eval ID: {record.evaluation_id} - Acc: {record.accuracy}")
        
    elif args.command == "promote":
        pm = PromotionManager(storage, events, PromotionPolicy())
        pd = pm.evaluate_promotion(args.global_model_id, args.eval_record_id)
        print(f"Promotion Decision: {pd.decision}")
        
if __name__ == "__main__":
    main()
