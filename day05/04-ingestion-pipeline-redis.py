print("----------------------直接查询 Redis 数据库----------------------")
import redis

redis_client = redis.Redis(host='127.0.0.1', port=6379, decode_responses=True, protocol=2)


def decode_value(value):
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return value

# 查看所有 keys
all_keys = redis_client.keys("my_test_cache*")
print(f"Redis 中的所有相关 keys: {len(all_keys)} 个")

# 查看前几个 key 的内容
for key in all_keys[:3]:
    value = redis_client.hgetall(key)
    print(f"Key: {decode_value(key)}")
    for k, v in value.items():
        # 将存储的内容进行转义
        text = decode_value(v)
        print(f"Value: {text.encode('utf-8').decode('unicode_escape')}")
    print("-" * 30)
