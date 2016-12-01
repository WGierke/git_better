from django.conf.urls import patterns, include, url
from django.contrib import admin


admin.autodiscover()

urlpatterns = patterns('',
                       url(r'^$', 'django_server.app.views.home'),
                       url(r'^admin/', include(admin.site.urls)),
                       url(r'^login/$', 'django_server.app.views.home'),
                       url(r'^logout/$', 'django_server.app.views.logout'),
                       url(r'^done/$', 'django_server.app.views.done', name='done'),
                       url(r'', include('social.apps.django_app.urls', namespace='social'))
                       )
