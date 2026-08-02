ACQUIRE = """
if redis.call('exists', KEYS[1]) == 0 then
    local token = redis.call('incr', KEYS[2])
    redis.call('psetex', KEYS[1], ARGV[2], ARGV[1] .. ':' .. token)
    return token
end
return 0
"""

VERIFY = """
if redis.call('get', KEYS[1]) == ARGV[1] then
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
local payload = redis.call('hget', KEYS[2], 'payload')
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
local current = tonumber(redis.call('hget', KEYS[2], 'version') or '0')
if current ~= tonumber(ARGV[2]) then
    return -1
end
redis.call('hset', KEYS[2], 'version', ARGV[3], 'payload', ARGV[4])
redis.call('pexpire', KEYS[2], ARGV[5])
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

