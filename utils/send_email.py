from airflow.utils.email import send_email


def notify_email(context) -> None:
    """Send custom email alerts."""

    # email title.
    title = "Airflow alert: {} Failed".format(context['task_instance'].task_id)

    # email contents
    body = """
    Hi Everyone, <br>
    <br>
    There's been an error in the {} job.<br>
    <br>
    Have fun debugging :),<br>
    Airflow bot <br>
    """.format(context['task_instance'].task_id
    )

    send_email("richjdowney@gmail.com", title, body)