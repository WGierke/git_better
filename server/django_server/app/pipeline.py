from social.pipeline.partial import partial


@partial
def save_auth_token(strategy, details, user=None, is_new=False, *args, **kwargs):
    """Save the GitHub access token for later use"""
    token = kwargs.get("response").get("access_token")
    user.token = token
    user.save()
