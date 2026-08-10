ACQUIRE = """
if redis.call('set', KEYS[1], ARGV[1], 'NX', 'PX', ARGV[2]) then
    return 1
end
return 0
"""

RENEW = """
if redis.call('get', KEYS[1]) == ARGV[1] then
    redis.call('pexpire', KEYS[1], ARGV[2])
    if redis.call('exists', KEYS[2]) == 1 then
        redis.call('pexpire', KEYS[2], ARGV[3])
    end
    return 1
end
return 0
"""

LOAD = """
if redis.call('get', KEYS[1]) ~= ARGV[1] then
    return {0}
end
local payload = redis.call('get', KEYS[2])
local active = redis.call('get', KEYS[3])
if payload then
    redis.call('pexpire', KEYS[2], ARGV[2])
end
if active then
    redis.call('pexpire', KEYS[3], ARGV[2])
end
return {1, payload or false, active or false}
"""

SAVE = """
if redis.call('get', KEYS[1]) ~= ARGV[1] then
    return 0
end
redis.call('psetex', KEYS[2], ARGV[3], ARGV[2])
return 1
"""

MARK_ACTIVE = """
if redis.call('get', KEYS[1]) ~= ARGV[1] then
    return 0
end
if redis.call('exists', KEYS[2]) == 1 then
    return -1
end
redis.call('psetex', KEYS[2], ARGV[3], ARGV[2])
return 1
"""

CHECKPOINT = """
if redis.call('get', KEYS[1]) ~= ARGV[1] then
    return 0
end
redis.call('psetex', KEYS[2], ARGV[3], ARGV[2])
redis.call('del', KEYS[3])
return 1
"""

RESET = """
if redis.call('get', KEYS[1]) ~= ARGV[1] then
    return 0
end
redis.call('del', KEYS[2], KEYS[3])
return 1
"""

RELEASE = """
if redis.call('get', KEYS[1]) == ARGV[1] then
    if redis.call('exists', KEYS[2]) == 1 then
        redis.call('pexpire', KEYS[2], ARGV[2])
    end
    redis.call('del', KEYS[1])
    return 1
end
return 0
"""
