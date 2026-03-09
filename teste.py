class Solution:
    def numberOfStableArrays(self, zero: int, one: int, limit: int) -> int:
        MOD = 10**9 + 7

        # dp[i][j][last]: number of stable arrays using i zeros and j ones,
        # ending with `last` (0 or 1)
        dp = [[[0, 0] for _ in range(one + 1)] for _ in range(zero + 1)]

        # Base cases: arrays of all zeros or all ones (up to limit length)
        for i in range(1, min(zero, limit) + 1):
            dp[i][0][0] = 1
        for j in range(1, min(one, limit) + 1):
            dp[0][j][1] = 1

        for i in range(1, zero + 1):
            for j in range(1, one + 1):
                # Ending with 0: extend a previous array by adding one more 0
                # dp[i-1][j][0] -> was already ending with 0, add another 0
                # dp[i-1][j][1] -> was ending with 1, switch to 0
                # Subtract overcounted: run of (limit+1) zeros
                dp[i][j][0] = (dp[i - 1][j][0] + dp[i - 1][j][1]) % MOD
                if i > limit:
                    # Remove cases where we'd have limit+1 consecutive 0s
                    # That means at position i-limit-1 zeros, j ones, it ended with 1
                    # and then we placed exactly limit+1 zeros
                    dp[i][j][0] = (dp[i][j][0] - dp[i - 1 - limit][j][1]) % MOD

                # Ending with 1: symmetric logic
                dp[i][j][1] = (dp[i][j - 1][0] + dp[i][j - 1][1]) % MOD
                if j > limit:
                    dp[i][j][1] = (dp[i][j][1] - dp[i][j - 1 - limit][0]) % MOD

        return (dp[zero][one][0] + dp[zero][one][1]) % MOD