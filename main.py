import numpy as np
import sys
input = sys.stdin.readline
print = sys.stdout.write

def initialize_parameters(a00,a01,a10,a11,pi0,pi1):
    A = np.array([[a00, a01], [a10, a11]])
    pi = np.array([pi0, pi1])
    return A, pi

def gaussian_pdf(x, mean, variance):
    if variance == 0:
        if x == mean:
            return 1
        else:
            return 0
    else:
        return (1 / np.sqrt(2 * np.pi * variance)) * np.exp(-(x - mean)**2 / (2 * variance))

def BW_Algo(returns, A, pi, mu, sigma_sq):
    T = len(returns)
    K = len(pi)
    alpha = np.zeros((T, K))
    beta = np.zeros((T, K))
    gamma = np.zeros((T, K))
    xi = np.zeros((T - 1, K, K))


    for i in range(0, K):
        alpha[0, i] = pi[i] * gaussian_pdf(returns[0], mu[i], sigma_sq[i])
    for t in range(1, T):
        for i in range(0, K):
            sum_alpha = 0
            for j in range(0, K):
                sum_alpha += alpha[t-1, j] * A[j, i]
            alpha[t, i] = sum_alpha * gaussian_pdf(returns[t], mu[i], sigma_sq[i])
    
    beta[T-1, 0] = 1
    beta[T-1, 1] = 1
    for t in range(T - 2, -1, -1):
        for i in range(0, K):
            sum_beta = 0
            for j in range(0, K):
                sum_beta += beta[t + 1, j] * A[i, j] * gaussian_pdf(returns[t + 1], mu[j], sigma_sq[j])
            beta[t, i] = sum_beta
    
    for t in range(0,T):
        for i in range(0,K):
            numerator = alpha[t,i] * beta[t, i]
            denominator = 0
            for j in range(K):
                denominator += alpha[t, j] * beta[t, j]
            if denominator == 0:
                gamma[t, i] = 0
            else:
                gamma[t, i] = numerator / denominator
    for t in range(0, T-1):
        for i in range(0, K):
            for j in range(0, K):
                numerator = alpha[t, i] * A[i, j] * beta[t+1, j] *gaussian_pdf(returns[t+1], mu[j], sigma_sq[j])
                denominator = 0
                for k in range(0, K):
                    for w in range(0, K):
                        denominator += alpha[t, k]*A[k, w]*beta[t+1, w]*gaussian_pdf(returns[t+1], mu[w], sigma_sq[w])
                if denominator == 0:
                    xi[t, i, j] = 0
                else:
                    xi[t, i, j] = numerator / denominator
    return gamma, xi

def update_parameters(returns, gamma, xi, mu_passed):
    T = len(returns)
    K = 2
    pi_star = np.zeros(K)
    for i in range(K):
        pi_star[i] = gamma[0, i]
    A_star = np.zeros((K, K))
    mu_star = np.zeros(K)
    sigma_sq_star = np.zeros(K)
    for i in range(K):
        for j in range(K):
            num_sum = 0
            den_sum = 0
            for t in range(T - 1):
                num_sum += xi[t, i, j]
                den_sum += gamma[t, i]
            if den_sum == 0:
                A_star[i, j] = 0
            else:
                A_star[i, j] = num_sum / den_sum

    for i in range(K):
        num_sum = 0
        den_sum = 0
        for t in range(T):
            num_sum += gamma[t,i]*returns[t]
            den_sum += gamma[t,i]
        if den_sum == 0:
            mu_star[i] = 0
        else:
            mu_star[i] = num_sum / den_sum
    
    for i in range(K):
        num_sum = 0
        den_sum = 0
        for t in range(T):
            num_sum += gamma[t,i]*((returns[t]-mu_passed[i])**2)
            den_sum += gamma[t,i]
        if den_sum == 0:
            sigma_sq_star[i] = 0
        else:
            sigma_sq_star[i] = num_sum / den_sum
    
    return pi_star, A_star, mu_star, sigma_sq_star

def converged(old_params, new_params, tol = 1e-8):
    if abs(np.linalg.norm(old_params - new_params)) <= tol:
        return True
    return False

def market_regimes(mu, posterior_prob):
    bullish = 0
    bearish = 1
    if mu[0]<mu[1]:
        bearish = 0
        bullish = 1
    regimes = []
    for prob in posterior_prob:
        if prob[bullish] > prob[bearish]:
            regimes.append("Bull")
        else:
            regimes.append("Bear")
    return regimes

if __name__ == "__main__":

    a00, a01, a10, a11 = map(float, input().split())
    pi0, pi1 = map(float, input().split())
    T = int(input())
    returns = [float(input()) for _ in range(T)]
    
    A, pi = initialize_parameters(a00, a01, a10, a11, pi0, pi1)
    mu = np.array([0, 0])
    sigma_sq = np.array([1, 1])

    posterior_prob = np.zeros((T, 2))

    for _ in range(100000):
        A_old, pi_old, mu_old, sigma_sq_old = A.copy(), pi.copy(), mu.copy(), sigma_sq.copy()
        gamma, xi = BW_Algo(returns, A, pi, mu, sigma_sq)
        pi, A, mu, sigma_sq = update_parameters(returns, gamma, xi, mu_old)
        if converged(A_old, A) and converged(pi_old, pi) and converged(mu_old, mu) and converged(sigma_sq_old, sigma_sq):
            posterior_prob = gamma
            break
    
    regimes = market_regimes(mu, posterior_prob)


    for regime in regimes:
        print(regime)
        print("\n")
