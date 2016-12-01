from django.shortcuts import redirect
from django.contrib.auth.decorators import login_required
from django.contrib.auth import logout as auth_logout

from django_server.app.decorators import render_to
from utils import build_repo_html


def logout(request):
    """Log user out"""
    auth_logout(request)
    return redirect('/')


def context(**extra):
    return dict({}, **extra)


@render_to('home.html')
def home(request):
    """Show home view, displays login mechanism"""
    if request.user.is_authenticated():
        return redirect('done')
    return context()


@login_required
@render_to('home.html')
def done(request):
    """Show login complete view, displays user data"""
    access_token = request.user.token
    try:
        repos_html = build_repo_html(access_token)
    except Exception, e:
        repos_html = "Can't show repositories: " + str(e)
    return context(repos_html=repos_html)
