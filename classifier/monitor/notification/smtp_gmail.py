import logging

from classifier.config.setting import monitor as cfg


@cfg.check(cfg.Notification, cfg.Gmail)
def gmail_send_text(title: str, body: str, subtype: str = "plain"):
    if not (cfg.Gmail.address and cfg.Gmail.password):
        return
    from email.mime.text import MIMEText

    msg = MIMEText(body, subtype)
    msg["Subject"] = title
    msg["From"] = cfg.Gmail.address
    msg["To"] = cfg.Gmail.address
    try:
        import smtplib

        with smtplib.SMTP_SSL(
            cfg.Gmail.smtp_server, cfg.Gmail.smtp_port
        ) as smtp_server:
            smtp_server.login(cfg.Gmail.address, cfg.Gmail.password)
            smtp_server.sendmail(
                cfg.Gmail.address, [cfg.Gmail.address], msg.as_string()
            )
    except Exception as e:
        logging.error(f"Failed to send email: {e}", exc_info=e)
