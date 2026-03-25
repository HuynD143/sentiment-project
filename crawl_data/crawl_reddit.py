import praw

class CrawlReddit:
    def __init__(self):
        self.reddit = praw.Reddit(
            client_id="CbC3aYc0Z5Vg10zLFZYhyA",
            client_secret="Yeuto2XmeEmAuEj_1-jKEeFlpTECuw",
            user_agent="python:sentiment-analysis-app:v1.0 (by /u/YOUR_REDDIT_USERNAME)"
        )

    def get_comments(self, url):
        self.submission = self.reddit.submission(url=url)
        self.submission.comments.replace_more(limit=0)
        # Lấy tuple (author, body)
        self.comments = [
            (comment.author.name if comment.author else "[deleted]", comment.body)
            for comment in self.submission.comments.list()
        ]
        return self.comments


# url = "https://www.reddit.com/r/HonkaiStarRail/comments/1oos0vz/do_not_press_this_button_it_has_spoilers/"
# reddit = CrawlReddit()
# comments = reddit.get_comments(url)
#
# for x in comments:
#     print(x[0] + "-----------" +  x[1])
#     exit(0)
