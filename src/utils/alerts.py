import os
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail

def send_alert(subject: str, message: str):
    """
    Send an email alert using SendGrid API.
    """
    api_key = os.getenv("SENDGRID_API_KEY")
    if not api_key:
        print("⚠️ No SENDGRID_API_KEY found in .env — skipping email alert.")
        return

    to_email = os.getenv("ALERT_EMAIL_TO")
    from_email = os.getenv("ALERT_EMAIL_FROM")

    mail = Mail(
        from_email=from_email,
        to_emails=to_email,
        subject=subject,
        html_content=message,
    )

    try:
        sg = SendGridAPIClient(api_key)
        response = sg.send(mail)
        print("✅ Alert email sent:", response.status_code)
    except Exception as e:
        print("❌ Error sending alert:", e)
