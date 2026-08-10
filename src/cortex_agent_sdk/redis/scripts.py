ACQUIRE = """
if redis.call('set', KEYS[1], ARGV[1], 'NX', 'PX', ARGV[2]) then
    return 1
end
return 0
"""

RENEW = """
if redis.call('get', KEYS[1]) == ARGV[1] then
    redis.call('pexpire', KEYS[1], ARGV[2])
    return 1
end
return 0
"""

LOAD = """
if redis.call('get', KEYS[1]) ~= ARGV[1] then
    return {0}
end
local payload = redis.call('get', KEYS[2])
if payload then
    redis.call('pexpire', KEYS[2], ARGV[2])
    return {1, payload}
end
return {1}
"""

SAVE = """
if redis.call('get', KEYS[1]) ~= ARGV[1] then
    return 0
end
redis.call('psetex', KEYS[2], ARGV[3], ARGV[2])
return 1
"""

RESET = """
if redis.call('get', KEYS[1]) ~= ARGV[1] then
    return 0
end
redis.call('del', KEYS[2])
return 1
"""

RELEASE = """
if redis.call('get', KEYS[1]) == ARGV[1] then
    redis.call('del', KEYS[1])
    return 1
end
return 0
"""
