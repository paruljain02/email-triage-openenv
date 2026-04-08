"""
Synthetic Email Generator — creates realistic corporate email scenarios.
Each email has ground truth labels for grading.
"""

import random
import hashlib
from datetime import datetime, timedelta
from models import (
    Email, EmailAttachment, ThreadMessage,
    EmailCategory, Priority, Sentiment, Department
)


# ── Templates ──────────────────────────────────────────────────────────────

EMAILS_DB = [
    # ── SUPPORT ────────────────────────────────────────────────────────
    {
        "sender": "maria.garcia@techstartup.io",
        "subject": "URGENT: Production database is down — all services affected",
        "body": (
            "Hi Team,\n\n"
            "Our production database cluster went down at 3:47 AM EST. All customer-facing "
            "services are currently returning 500 errors. We've tried restarting the primary "
            "node but it's stuck in recovery mode.\n\n"
            "This is affecting approximately 15,000 active users. Our SLA breach window is "
            "in 2 hours. We need immediate help from the database team.\n\n"
            "Error logs attached. Please escalate to P0 immediately.\n\n"
            "Maria Garcia\nSRE Lead, TechStartup"
        ),
        "ground_truth": {
            "category": "urgent",
            "sentiment": "negative",
            "priority": "P0",
            "department": "engineering",
            "action_items": [
                "Escalate to database team immediately",
                "Review attached error logs",
                "Assess SLA breach timeline"
            ],
            "requires_follow_up": True,
            "reply_tone": "urgent, acknowledging severity, committing to immediate action"
        },
        "has_attachments": True,
        "attachments": [{"filename": "db_error_logs_20241208.txt", "file_type": "text/plain", "size_kb": 245}],
        "difficulty": "easy_to_classify"
    },
    {
        "sender": "john.smith@megacorp.com",
        "subject": "Re: Invoice #INV-2024-0892 — payment discrepancy",
        "body": (
            "Hello,\n\n"
            "I've reviewed invoice #INV-2024-0892 and there's a discrepancy. We were quoted "
            "$45,000 for the annual license but the invoice shows $52,500. That's a $7,500 "
            "difference that nobody on our end approved.\n\n"
            "Additionally, the invoice lists 'Premium Support Tier' which we never agreed to. "
            "Our contract clearly states Standard Support.\n\n"
            "Please correct this before our AP department processes it. We need a revised "
            "invoice by end of week.\n\n"
            "Best regards,\nJohn Smith\nProcurement Manager, MegaCorp"
        ),
        "ground_truth": {
            "category": "billing",
            "sentiment": "negative",
            "priority": "P1",
            "department": "finance",
            "action_items": [
                "Review invoice #INV-2024-0892 against contract",
                "Verify support tier in contract",
                "Issue corrected invoice by end of week"
            ],
            "requires_follow_up": True,
            "reply_tone": "apologetic, professional, acknowledging the error, committing to correction"
        },
        "is_reply": True,
        "thread_history": [
            {"sender": "billing@ourcompany.com", "body": "Please find attached invoice #INV-2024-0892 for your annual renewal.", "timestamp": "2024-12-05T09:00:00Z"}
        ],
        "difficulty": "medium"
    },
    {
        "sender": "priya.patel@innovateai.com",
        "subject": "Partnership proposal — AI integration opportunity",
        "body": (
            "Dear Team,\n\n"
            "I'm the VP of Business Development at InnovateAI. We've been following your "
            "company's growth and believe there's a strong synergy between our AI capabilities "
            "and your platform.\n\n"
            "We'd love to explore a partnership where we integrate our NLP engine into your "
            "product suite. Our technology currently serves 200+ enterprise clients and has "
            "shown 40% improvement in workflow automation.\n\n"
            "Would you be open to a 30-minute call next week? I've attached our partnership "
            "one-pager and a technical overview.\n\n"
            "Looking forward to connecting.\n\n"
            "Best,\nPriya Patel\nVP Business Development, InnovateAI"
        ),
        "ground_truth": {
            "category": "partnership",
            "sentiment": "positive",
            "priority": "P2",
            "department": "sales",
            "action_items": [
                "Review partnership one-pager",
                "Schedule 30-min exploratory call",
                "Loop in product team for technical assessment"
            ],
            "requires_follow_up": True,
            "reply_tone": "warm, professional, expressing interest, suggesting next steps"
        },
        "has_attachments": True,
        "attachments": [
            {"filename": "InnovateAI_Partnership_OnePager.pdf", "file_type": "application/pdf", "size_kb": 890},
            {"filename": "Technical_Overview.pdf", "file_type": "application/pdf", "size_kb": 1250}
        ],
        "difficulty": "medium"
    },
    {
        "sender": "noreply@win-prize-now-2024.xyz",
        "subject": "🎉 CONGRATULATIONS! You've Won $1,000,000!!! Click NOW!!!",
        "body": (
            "DEAR LUCKY WINNER,\n\n"
            "You have been SELECTED as the GRAND PRIZE WINNER of our International Lottery!\n\n"
            "YOUR PRIZE: $1,000,000 USD!!!\n\n"
            "To claim your prize, simply click the link below and provide your bank details:\n"
            "http://totally-legit-prize.xyz/claim?id=abc123\n\n"
            "ACT NOW! This offer expires in 24 hours!\n\n"
            "CONGRATULATIONS AGAIN!\n"
            "International Prize Committee"
        ),
        "ground_truth": {
            "category": "spam",
            "sentiment": "neutral",
            "priority": "P3",
            "department": "spam_filter",
            "action_items": ["Mark as spam and block sender domain"],
            "requires_follow_up": False,
            "reply_tone": "none — do not reply"
        },
        "difficulty": "easy_to_classify"
    },
    {
        "sender": "lisa.wong@ourcompany.com",
        "subject": "Q4 OKR review — preparation needed",
        "body": (
            "Hi everyone,\n\n"
            "Quick reminder that our Q4 OKR review is scheduled for Friday at 2 PM. "
            "Please have the following ready:\n\n"
            "1. Updated progress on your key results (percentage complete)\n"
            "2. Any blockers or risks that need escalation\n"
            "3. Proposed OKRs for Q1 next year\n\n"
            "I'll send out the review template by EOD tomorrow. If you can't make Friday, "
            "please let me know ASAP so we can adjust.\n\n"
            "Thanks,\nLisa Wong\nChief of Staff"
        ),
        "ground_truth": {
            "category": "internal",
            "sentiment": "neutral",
            "priority": "P2",
            "department": "executive",
            "action_items": [
                "Prepare Q4 OKR progress update",
                "Identify blockers for escalation",
                "Draft proposed Q1 OKRs",
                "Confirm attendance for Friday 2 PM"
            ],
            "requires_follow_up": True,
            "reply_tone": "professional, confirming attendance, acknowledging preparation items"
        },
        "difficulty": "easy_to_classify"
    },
    {
        "sender": "alex.chen@bigclient.com",
        "subject": "Considering switching to your competitor — last chance",
        "body": (
            "Hi,\n\n"
            "I'll be blunt. We've been a customer for 3 years paying $120K/year, and the "
            "experience has been declining. In the last quarter alone:\n\n"
            "- 4 major outages affecting our team\n"
            "- Support tickets taking 72+ hours for first response\n"
            "- Features we were promised on the roadmap still missing\n\n"
            "CompetitorX has approached us with a very competitive offer. Before we make a "
            "decision, I wanted to give you one last chance to make this right.\n\n"
            "I need a call with someone senior by Thursday, or we're moving forward with "
            "the switch.\n\n"
            "Alex Chen\nCTO, BigClient"
        ),
        "ground_truth": {
            "category": "urgent",
            "sentiment": "angry",
            "priority": "P0",
            "department": "executive",
            "action_items": [
                "Escalate to VP/C-level immediately",
                "Schedule senior leadership call before Thursday",
                "Prepare retention package and service recovery plan",
                "Review account history and recent incidents"
            ],
            "requires_follow_up": True,
            "reply_tone": "deeply empathetic, acknowledging failures, committing to executive-level attention"
        },
        "difficulty": "hard"
    },
    {
        "sender": "recruiter@talentfirm.com",
        "subject": "Exciting opportunity — Senior ML Engineer role",
        "body": (
            "Hi there,\n\n"
            "I came across your profile and wanted to reach out about a Senior ML Engineer "
            "position at a top-tier AI company (Series C, $2B valuation).\n\n"
            "Compensation: $250K-$350K + equity\nLocation: Remote-first\n"
            "Tech stack: PyTorch, Kubernetes, Ray\n\n"
            "Would you be open to a quick 15-minute chat to learn more?\n\n"
            "Best,\nJamie\nSenior Recruiter, TalentFirm"
        ),
        "ground_truth": {
            "category": "hr",
            "sentiment": "positive",
            "priority": "P3",
            "department": "hr",
            "action_items": ["Forward to HR/recruiting team if relevant, otherwise archive"],
            "requires_follow_up": False,
            "reply_tone": "polite acknowledgment or decline"
        },
        "difficulty": "easy_to_classify"
    },
    {
        "sender": "sarah.johnson@ourcompany.com",
        "subject": "CONFIDENTIAL: Potential data breach detected",
        "body": (
            "CONFIDENTIAL — DO NOT FORWARD\n\n"
            "Team,\n\n"
            "Our security monitoring system flagged unusual data access patterns at 11:32 PM "
            "last night. Preliminary analysis suggests:\n\n"
            "- 3 internal accounts accessed customer PII tables outside normal hours\n"
            "- ~50,000 records were exported via API\n"
            "- The access originated from an IP in a region where we have no employees\n\n"
            "I've initiated our incident response protocol but need legal and engineering "
            "involved ASAP. We may have regulatory notification obligations within 72 hours.\n\n"
            "Please treat this as highest priority.\n\n"
            "Sarah Johnson\nCISO"
        ),
        "ground_truth": {
            "category": "urgent",
            "sentiment": "negative",
            "priority": "P0",
            "department": "legal",
            "action_items": [
                "Engage legal team for regulatory obligations assessment",
                "Loop in engineering for forensic investigation",
                "Review 72-hour notification timeline",
                "Restrict flagged account access immediately",
                "Schedule incident response war room"
            ],
            "requires_follow_up": True,
            "reply_tone": "serious, acknowledging severity, confirming immediate engagement"
        },
        "difficulty": "hard"
    },
    {
        "sender": "david.miller@vendor.com",
        "subject": "Annual license renewal — early bird discount",
        "body": (
            "Hi,\n\n"
            "Your annual license for CloudTools Enterprise is up for renewal on January 15th. "
            "As a valued customer, we'd like to offer you our early bird discount:\n\n"
            "- Renew by Dec 31: 15% discount ($38,250 instead of $45,000)\n"
            "- Renew by Jan 10: 10% discount ($40,500)\n"
            "- After Jan 15: Standard pricing applies\n\n"
            "Shall I prepare the renewal paperwork? Happy to answer any questions.\n\n"
            "Regards,\nDavid Miller\nAccount Executive, CloudTools"
        ),
        "ground_truth": {
            "category": "sales",
            "sentiment": "positive",
            "priority": "P2",
            "department": "finance",
            "action_items": [
                "Evaluate early bird discount timeline",
                "Check budget approval for renewal",
                "Decide on renewal tier by Dec 31 deadline"
            ],
            "requires_follow_up": True,
            "reply_tone": "professional, acknowledging offer, asking for time to review internally"
        },
        "difficulty": "medium"
    },
    {
        "sender": "marketing@randomnewsletter.com",
        "subject": "🚀 Top 10 Growth Hacks for 2025 — Free Webinar Inside!",
        "body": (
            "Hey there!\n\n"
            "We're hosting an EXCLUSIVE free webinar on the top growth strategies for 2025!\n\n"
            "📅 Date: Dec 20, 2024\n🕐 Time: 1 PM EST\n\n"
            "Featured speakers include growth leaders from Stripe, Notion, and Figma.\n\n"
            "Reserve your spot now → [Register Here]\n\n"
            "Can't make it? We'll send you the recording!\n\n"
            "Cheers,\nThe GrowthPulse Team\n\n"
            "Unsubscribe | Privacy Policy"
        ),
        "ground_truth": {
            "category": "spam",
            "sentiment": "neutral",
            "priority": "P3",
            "department": "spam_filter",
            "action_items": ["Archive or unsubscribe"],
            "requires_follow_up": False,
            "reply_tone": "none"
        },
        "difficulty": "easy_to_classify"
    },
    {
        "sender": "emma.davis@ourcompany.com",
        "subject": "Re: Parental leave policy question",
        "body": (
            "Hi HR Team,\n\n"
            "I'm expecting in March and wanted to understand our parental leave policy. "
            "Specifically:\n\n"
            "1. How many weeks of paid leave are available?\n"
            "2. Can I split the leave (e.g., some before and some after)?\n"
            "3. Is there a gradual return-to-work option?\n"
            "4. How does this interact with our short-term disability insurance?\n\n"
            "I'd appreciate if we could schedule a private meeting to discuss. "
            "Preferably before the holiday break.\n\n"
            "Thank you,\nEmma Davis\nSenior Product Manager"
        ),
        "ground_truth": {
            "category": "hr",
            "sentiment": "neutral",
            "priority": "P2",
            "department": "hr",
            "action_items": [
                "Schedule private meeting with Emma before holiday break",
                "Prepare parental leave policy documentation",
                "Review disability insurance interaction",
                "Check flexible return-to-work options"
            ],
            "requires_follow_up": True,
            "reply_tone": "warm, supportive, professional, confirming meeting setup"
        },
        "difficulty": "medium"
    },
    {
        "sender": "support-ticket-4521@helpdesk.ourcompany.com",
        "subject": "[Ticket #4521] Customer unable to login after password reset",
        "body": (
            "New support ticket from: rachel.torres@smallbiz.com\n\n"
            "Description:\n"
            "I reset my password 3 times today but still can't log in. I keep getting "
            "'Invalid credentials' error even though I JUST set the password. I've tried "
            "different browsers and clearing cookies. Nothing works.\n\n"
            "I have a demo with a client in 1 hour and I desperately need access to my "
            "dashboard. PLEASE HELP!\n\n"
            "Account: rachel.torres@smallbiz.com\nPlan: Business Pro\n"
            "Browser: Chrome 120.0\nOS: macOS Sonoma 14.2"
        ),
        "ground_truth": {
            "category": "support",
            "sentiment": "negative",
            "priority": "P1",
            "department": "support",
            "action_items": [
                "Check authentication logs for the account",
                "Verify password reset flow is working correctly",
                "Provide temporary access or manual password reset",
                "Respond within 30 minutes given client demo urgency"
            ],
            "requires_follow_up": True,
            "reply_tone": "empathetic, urgent, providing immediate troubleshooting steps"
        },
        "difficulty": "medium"
    },
]


def generate_email_batch(
    count: int = 10,
    seed: int = 42,
    difficulty_filter: str = None
) -> list[dict]:
    """
    Generate a batch of emails with ground truth labels.
    Returns list of dicts with 'email' (Email model) and 'ground_truth'.
    """
    random.seed(seed)

    pool = EMAILS_DB.copy()
    if difficulty_filter:
        pool = [e for e in pool if difficulty_filter in e.get("difficulty", "")]

    # If requesting more than pool size, cycle through
    selected = []
    for i in range(count):
        template = pool[i % len(pool)]
        sender = template["sender"]
        email_id = f"email_{hashlib.md5(f'{sender}_{i}_{seed}'.encode()).hexdigest()[:8]}"

        attachments = []
        if template.get("has_attachments"):
            attachments = [EmailAttachment(**a) for a in template.get("attachments", [])]

        thread_history = []
        if template.get("thread_history"):
            thread_history = [ThreadMessage(**t) for t in template["thread_history"]]

        email = Email(
            id=email_id,
            sender=template["sender"],
            sender_domain=template["sender"].split("@")[1],
            recipient="inbox@ourcompany.com",
            subject=template["subject"],
            body=template["body"],
            timestamp=(datetime(2024, 12, 8, 8, 0) + timedelta(minutes=i * 15)).isoformat() + "Z",
            has_attachments=template.get("has_attachments", False),
            attachments=attachments,
            thread_history=thread_history,
            is_reply=template.get("is_reply", False),
            cc=template.get("cc", []),
            metadata={"source_index": i}
        )

        selected.append({
            "email": email,
            "ground_truth": template["ground_truth"],
            "difficulty": template.get("difficulty", "medium")
        })

    random.shuffle(selected)
    return selected
