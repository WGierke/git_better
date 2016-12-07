import os

if __name__ == '__main__':
    PORT = os.environ.get("PORT", 8000)
    os.system('python server/manage.py runserver 0.0.0.0:{}'.format(PORT))
