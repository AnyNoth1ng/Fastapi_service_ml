import redis


r = redis.Redis(host='localhost', port=6379, decode_responses=True)
# r.set('123', 124)
print(r.get('123'))

