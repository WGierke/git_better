import Queue
import threading
from tqdm import tqdm
import pandas as pd
import requests
from app.utils import website_exists


def update_df(path="data/processed_data.csv"):
    THREADS_COUNT = 5
    data_frame = pd.DataFrame.from_csv(path)
    bar = tqdm(total=len(data_frame))
    df_q = Queue.LifoQueue()
    df_q.put(data_frame)
    token_q = Queue.LifoQueue()
    for i in range(THREADS_COUNT):
        token_q.put(i)

    threads = []
    for index, row in data_frame.iterrows():
        t = threading.Thread(target=update_features, args=(index, row, bar, df_q, token_q))
        t.daemon = True
        threads.append(t)

    for t in threads:
        t.start()

    for t in threads:
        t.join()

    df_q.get().to_csv(path)


def update_features(index, row, bar, df_q, token_q):
    """Define which features should be updated here. Preferably without using the Github API since we have no token renewal, yet."""
    owner = row['owner']
    name = row['name']

    token = token_q.get()
    try:
        new_data_frame = pd.DataFrame.from_dict(row).T
        is_owner_homepage = name.lower() == "{}.github.io".format(owner.lower()) or name.lower() == "{}.github.com".format(owner.lower())
        has_homepage = website_exists("http://{}.github.io/{}".format(owner, name))
        has_license = "octicon octicon-law" in requests.get("https://github.com/{}/{}".format(owner, name)).text
        has_travis_config = website_exists("https://github.com/{}/{}/blob/master/.travis.yml".format(owner, name), only_headers=True)
        has_circle_config = website_exists("https://github.com/{}/{}/blob/master/circle.yml".format(owner, name), only_headers=True)
        has_ci_config = has_travis_config or has_circle_config

        new_data_frame.set_value(index, "isOwnerHomepage", is_owner_homepage)
        new_data_frame.set_value(index, "hasHomepage", has_homepage)
        new_data_frame.set_value(index, "hasLicense", has_license)
        new_data_frame.set_value(index, "hasTravisConfig", has_travis_config)
        new_data_frame.set_value(index, "hasCircleConfig", has_circle_config)
        new_data_frame.set_value(index, "hasCiConfig", has_ci_config)
    except Exception, e:
        print "Exception in aggregate_features: " + str(e)
        token_q.put(token)
        return

    token_q.put(token)
    shared_data_frame = df_q.get()
    update_columns = [col for col in ["isOwnerHomepage", "hasHomepage", "hasLicense", "hasTravisConfig", "hasCircleConfig", "hasCiConfig"]]
    for col in update_columns:
        try:
            shared_data_frame.set_value(index, col, new_data_frame.loc[index, col])
        except Exception, e:
            print "An error occured while fetching {}/{} and setting {}: {}".format(owner, name, col, e)
    df_q.put(shared_data_frame)
    bar.update()


if __name__ == '__main__':
    update_df()
