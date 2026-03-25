import googleapiclient.discovery
import pandas as pd
import time
from googleapiclient.errors import HttpError
import os
from dotenv import load_dotenv
import re
class Crawler:
    def __init__(self, url):
        load_dotenv()
        self.API_KEY = os.environ.get("API_KEY")
        self.VIDEO_ID = self.get_video_id_from_url(url)
        self.youtube = googleapiclient.discovery.build(
            'youtube',
            'v3',
            developerKey=self.API_KEY,
            cache_discovery=False
        )
    def get_youtube_comments(self):
        self.comments = []
        next_page_token = None
        try:
            while True:
                request = self.youtube.commentThreads().list(
                    part='snippet',
                    videoId=self.VIDEO_ID,
                    maxResults=100,
                    pageToken=next_page_token,
                    textFormat='plainText'
                )
                response = request.execute()
                if not response['items']:
                    break
                for item in response['items']:
                    comment_data = item['snippet']['topLevelComment']['snippet']
                    self.comments.append({
                        'author': comment_data['authorDisplayName'],
                        'text': comment_data['textDisplay'],
                        'likes': comment_data['likeCount'],
                        'published_at': comment_data['publishedAt'],
                        'updated_at': comment_data['updatedAt']
                    })
                next_page_token = response.get('nextPageToken')
                if not next_page_token:
                    break
        except HttpError as e:
            if e.resp.status == 403:
                print("Lỗi 403")
            elif e.resp.status == 404:
                print("Lỗi 404")
            else:
                print(f"Lỗi HTTP {e.resp.status}: {e}")
        except Exception as e:
            print(f"Lỗi: {e}")
        return self.comments
    def save_cmt(self):
        if self.comments:
            df = pd.DataFrame(self.comments)
            df.to_csv('youtube_comments.csv', index=False, encoding='utf-8')
            print(f"{len(self.comments)} comments")
            for i, comment in enumerate(self.comments, 1):
                print(f"{i}. {comment['author']}: {comment['text']}")
    def get_video_id_from_url(self, url):
        regex = r'(?<=v=)[a-zA-Z0-9_-]{11}'
        video_id = re.search(regex, url).group()
        print(video_id)
        return video_id
# crawl = Crawler('https://www.youtube.com/watch?v=XIWANvgXtvI')