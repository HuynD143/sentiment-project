import googleapiclient.discovery
import pandas as pd
import time
from googleapiclient.errors import HttpError
import os
from dotenv import load_dotenv
import re
import asyncio

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
        self.comments = []
    def get_youtube_comments(self):
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
    async def save_cmt(self, url='./youtube_comments.csv'):
        if self.comments:
            df = pd.DataFrame(self.comments)
            df.to_csv(f"{url}", index=False, encoding='utf-8')
            print(f"{len(self.comments)} comments")
            for i, comment in enumerate(self.comments, 1):
                print(f"{i}. {comment['author']}: {comment['text']}")

    def get_video_id_from_url(self, url):
        patterns = [
            r'(?<=v=)[a-zA-Z0-9_-]{11}',  # youtube.com/watch?v=
            r'youtu\.be/([a-zA-Z0-9_-]{11})',  # youtu.be/...
            r'youtube\.com/shorts/([a-zA-Z0-9_-]{11})'  # youtube.com/shorts/...
        ]
        for p in patterns:
            match = re.search(p, url)
            if match:
                # Nếu regex có group, lấy group(1), còn không lấy group()
                if match.lastindex:  # lastindex = số lượng nhóm match
                    return match.group(1)
                else:
                    return match.group()
        raise ValueError(f"❌ Không tìm thấy video ID trong URL: {url}")

# c = Crawler(url='https://youtu.be/APwGEtl7lcg?si=YAxQoopTJS5ugnX3')
# c.get_youtube_comments()
# # print(c.comments)
# content = [c['text'] for c in c.comments if isinstance(c, dict) and 'text' in c]
# authors = [c['author'] for c in c.comments if isinstance(c, dict) and 'author' in c]
# for x, y in zip(authors, content):
#     print(x + ": " + y)
