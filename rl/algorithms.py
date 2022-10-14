import numpy as np
from tqdm.notebook import tqdm
from rl.concurrency import concurrent_runs
from math import ceil, log

def comparison(environment, algorithm, runs, episodes, alpha_values, line_values, *args, seed=None, processes=None, show_progress_bar=True, bar_desc=None, leave_bar=False):
    results = np.full(shape=(len(line_values),len(alpha_values)), fill_value = None).tolist()
    with tqdm(total=len(alpha_values)*len(line_values)) as bar:
        for i,val in enumerate(line_values):
            for j,alpha in enumerate(alpha_values):
                l = concurrent_runs(runs, algorithm, val, alpha, episodes, environment, *args, seed=seed, processes=processes, show_progress_bar=show_progress_bar, bar_desc=bar_desc, leave_bar=leave_bar)
                results[i][j] = l
                bar.update(1)
    return results

def comparison_epsilon(environment, algorithm, runs, episodes, alpha, parameter, epsilon_values,  *args, seed=None, processes=None, show_progress_bar=True, bar_desc=None, leave_bar=False):
    results = np.full(shape=epsilon_values.shape, fill_value = None).tolist()
    with tqdm(total=epsilon_values.size) as bar:
        for i,val in enumerate(epsilon_values):
            runs, algorithm, *args
            l = concurrent_runs(runs, algorithm, parameter, alpha, episodes, environment, val, *args, seed=seed, processes=processes, show_progress_bar=show_progress_bar, bar_desc=bar_desc, leave_bar=leave_bar)
            results[i] = l
            bar.update(1)
    return results

def nstepTD(n, alpha, episodes, environment_class, env_n, gamma, value_function_class, initial_value, aggr_groups, alpha_strategy, single_update, to_return, seed):
    """
    n-step TD algorithm

    args:
        episodes: number of episodes
        environment_class: class of the environment
        env_n: size or name of the environment
        n: value of n
        alpha: initial step size
        gamma: discount factor
        value_function_class: class of the state value function
        initial_value: initial value of the value function
        aggr_groups: number of groups for state aggregation
        alpha_strategy: strategy for the step size - fixed, inverse_ep, incremental_mean
        single_update: accumulate the updates and do a single update at the end of the episode
        to_return: object to return - value_function, avg_rmse_ep, rmse_ep, rmse_step, last_rmse_ep, ep_length, avg_ep_length
        seed: numpy seed for reproducibility
    """
    env = environment_class(env_n, save_current_episode=True, seed=seed)
    value_function = value_function_class(env, initial_value, gamma, aggr_groups)
    initial_rmse = value_function.calculate_rmse()
    results = []
    gamma_powers = np.power(gamma, np.arange(0, n+1))
    visits = np.ones(value_function.shape)
    rmse = 'rmse_' in to_return
    length = '_length' in to_return
    avg = 'avg_' in to_return
    ep = 'ep' in to_return
    step = '_step' in to_return
    incremental_mean = alpha_strategy == 'incremental_mean'
    for episode in range(episodes):
        if not overflow:
            n_ep = n
            alpha_ep = alpha/(episode+1) if alpha_strategy=='inverse_ep' else alpha
            # Run episode
            env.reset()
            T = np.inf
            t = 0
            while t+1-n_ep<=T-1:
                if not env.done:
                    # Take an action
                    state, reward = env.step()
                    if env.done:
                        T = t+1
                        n_ep = min(n_ep,T)# with n greater than T we would have same returns (slightly speeds up the algorithm)
                tau = t+1-n_ep
                if 0<=tau<=T-1:
                    G = value_function.calculate_return(tau, n_ep, T, gamma_powers)
                    state_index = value_function.get_index(env.episode[tau][1])
                    delta = G - value_function.get_value(env.episode[tau][1])
                    if single_update:
                        value_function.save_update_state(env.episode[tau][1], alpha_ep/visits[state_index] * delta)
                    else:
                        value_function.update_state(env.episode[tau][1], alpha_ep/visits[state_index] * delta)
                    if incremental_mean:
                        visits[state_index] += 1.0
                    # Avoid overflow
                    if abs(value_function.get_value_by_index(state_index))>max_value*10000:
                        overflow = True
                        break
                if step:
                    results.append(value_function.calculate_rmse())
                t += 1
            if single_update:
                value_function.update_saved()
        if ep:
            if length:
                results.append(env.episode_length)
            else:
                results.append(value_function.calculate_rmse())
    if to_return == 'value_function':
        return value_function.w
    if to_return == 'last_rmse_ep':
        return results[-1]
    if avg:
        return np.array(results).mean()  
    if rmse:
        results.insert(0, initial_rmse)
    return results

def nstepTD_rec_wrong(n, alpha, episodes, environment_class, env_n, gamma, value_function_class, initial_value, aggr_groups, alpha_strategy, single_update, to_return, seed):
    """
    n-step TD algorithm with recursive calculation of returns with implementation error (accumulation of round-off errors)

    args:
        episodes: number of episodes
        environment_class: class of the environment
        env_n: size or name of the environment 
        n: value of n
        alpha: initial step size
        gamma: discount factor
        value_function_class: class of the state value function
        initial_value: initial value of the value function
        aggr_groups: number of groups for state aggregation
        alpha_strategy: strategy for the step size - fixed, inverse_ep, incremental_mean
        single_update: accumulate the updates and do a single update at the end of the episode
        to_return: object to return - value_function, avg_rmse_ep, rmse_ep, rmse_step, last_rmse_ep, ep_length, avg_ep_length
        seed: numpy seed for reproducibility
    """
    
    env = environment_class(env_n, save_current_episode=True, seed=seed)
    value_function = value_function_class(env, initial_value, gamma, aggr_groups)
    max_value = np.max(np.abs(value_function.true_value_function))
    initial_rmse = value_function.calculate_rmse()
    results = []
    visits = np.ones(value_function.shape)
    rmse = 'rmse_' in to_return
    length = '_length' in to_return
    avg = 'avg_' in to_return
    ep = 'ep' in to_return
    step = '_step' in to_return
    incremental_mean = alpha_strategy == 'incremental_mean'
    overflow = False
    for episode in range(episodes):
        if not overflow:
            n_ep = n
            alpha_ep = alpha/(episode+1) if alpha_strategy=='inverse_ep' else alpha
            # Run episode
            state = env.reset()
            T = np.inf
            t = 0
            old_v = 0.0
            G = 0.0
            gamma_power = 1.0
            while t+1-n_ep<=T-1:
                delta_p = 0.0
                if not env.done:
                    # Take an action
                    state, reward = env.step()
                    new_v = 0.0 if env.done else value_function.get_value(state)
                    if env.done:
                        T = t+1
                        n_ep = min(n_ep,T)# with n greater than T we would have same returns (slightly speeds up the algorithm) 
                    delta_p = reward+gamma*new_v-old_v
                    old_v = new_v
                if T-t>=1:
                    G += gamma_power*delta_p
                    if t<n_ep-1:
                        gamma_power *= gamma
                tau = t+1-n_ep
                if 0<=tau<=T-1:
                    state_index = value_function.get_index(env.episode[tau][1])
                    delta = G - value_function.get_value(env.episode[tau][1])
                    if single_update:
                        value_function.save_update_state(env.episode[tau][1], alpha_ep/visits[state_index] * delta)
                    else:
                        value_function.update_state(env.episode[tau][1], alpha_ep/visits[state_index] * delta)
                    if incremental_mean:
                        visits[state_index] += 1.0
                    G = (G-env.episode[tau+1][0])/gamma
                    # Avoid overflow
                    if abs(value_function.get_value_by_index(state_index))>max_value*10000:
                        overflow = True
                        break
                if step:
                    results.append(value_function.calculate_rmse())
                t += 1
            if single_update:
                value_function.update_saved()
        if ep:
            if length:
                results.append(env.episode_length)
            else:
                results.append(value_function.calculate_rmse())
    if to_return == 'value_function':
        return value_function.w
    if to_return == 'last_rmse_ep':
        return results[-1]
    if avg:
        return np.array(results).mean() 
    if rmse:
        results.insert(0, initial_rmse)
    return results

def nstepTD_rec(n, alpha, episodes, environment_class, env_n, gamma, value_function_class, initial_value, aggr_groups, alpha_strategy, single_update, to_return, seed):
    """
    n-step TD algorithm with recursive calculation of returns

    args:
        episodes: number of episodes
        environment_class: class of the environment
        env_n: size or name of the environment 
        n: value of n
        alpha: initial step size
        gamma: discount factor
        value_function_class: class of the state value function
        initial_value: initial value of the value function
        aggr_groups: number of groups for state aggregation
        alpha_strategy: strategy for the step size - fixed, inverse_ep, incremental_mean
        single_update: accumulate the updates and do a single update at the end of the episode
        to_return: object to return - value_function, avg_rmse_ep, rmse_ep, rmse_step, last_rmse_ep, ep_length, avg_ep_length
        seed: numpy seed for reproducibility
    """
    env = environment_class(env_n, save_current_episode=True, seed=seed)
    value_function = value_function_class(env, initial_value, gamma, aggr_groups)
    max_value = np.max(np.abs(value_function.true_value_function))
    initial_rmse = value_function.calculate_rmse()
    results = []
    gamma_last_power = np.power(gamma, n-1)
    visits = np.ones(value_function.shape)
    rmse = 'rmse_' in to_return
    length = '_length' in to_return
    avg = 'avg_' in to_return
    ep = 'ep' in to_return
    step = '_step' in to_return
    incremental_mean = alpha_strategy == 'incremental_mean'
    overflow = False
    for episode in range(episodes):
        if not overflow:
            n_ep = n
            alpha_ep = alpha/(episode+1) if alpha_strategy=='inverse_ep' else alpha
            # Run episode
            state = env.reset()
            T = np.inf
            t = 0
            old_v = 0.0
            G_calc = 0.0
            G = None
            i = 0
            gamma_power = 1.0
            while t+1-n_ep<=T-1:
                if not env.done:
                    # Take an action
                    state, reward = env.step()
                    new_v = 0.0 if env.done else value_function.get_value(state)
                    delta_p = reward+gamma*new_v-old_v
                    old_v = new_v
                    G_calc += gamma_power*delta_p
                    if i==n_ep-1:# n-step return calculated -> start calculating a new return
                        G = G_calc
                        G_calc = old_v
                        gamma_power = 1.0
                        i = 0
                    else:
                        i+=1
                        gamma_power *= gamma
                    if env.done:
                        T = t+1
                        if i>0:
                            G = G_calc if G is None else G+gamma_last_power*delta_p
                        n_ep = min(n_ep,T)# with n greater than T we would have same returns (slightly speeds up the algorithm)
                    elif G is not None and i>0: 
                        G += gamma_last_power*delta_p
                tau = t+1-n_ep
                if 0<=tau<=T-1:
                    state_index = value_function.get_index(env.episode[tau][1])
                    delta = G - value_function.get_value(env.episode[tau][1])
                    if single_update:
                        value_function.save_update_state(env.episode[tau][1], alpha_ep/visits[state_index] * delta)
                    else:
                        value_function.update_state(env.episode[tau][1], alpha_ep/visits[state_index] * delta)
                    if incremental_mean:
                        visits[state_index] += 1.0
                    if n_ep>1:
                        G = (G-env.episode[tau+1][0])/gamma
                    # Avoid overflow
                    if abs(value_function.get_value_by_index(state_index))>max_value*10000:
                        overflow = True
                        break
                if step:
                    results.append(value_function.calculate_rmse())
                t += 1
            if single_update:
                value_function.update_saved()
        if ep:
            if length:
                results.append(env.episode_length)
            else:
                results.append(value_function.calculate_rmse())
    if to_return == 'value_function':
        return value_function.w
    if to_return == 'last_rmse_ep':
        return results[-1]
    if avg:
        return np.array(results).mean() 
    if rmse:
        results.insert(0, initial_rmse)
    return results

def λreturn(_lambda, alpha, episodes, environment_class, env_n, gamma, value_function_class, initial_value, aggr_groups, alpha_strategy, single_update, to_return, seed):
    """
    λ-return algorithm

    args:
        episodes: number of episodes
        environment_class: class of the environment
        env_n: size or name of the environment
        _lambda: value of lambda
        alpha: initial step size
        gamma: discount factor
        value_function_class: class of the state value function
        initial_value: initial value of the value function
        aggr_groups: number of groups for state aggregation
        alpha_strategy: strategy for the step size - fixed, inverse_ep, incremental_mean
        single_update: accumulate the updates and do a single update at the end of the episode
        to_return: object to return - value_function, avg_rmse_ep, rmse_ep, rmse_step, ep_length, avg_ep_length
        seed: numpy seed for reproducibility
    """
    env = environment_class(env_n, save_current_episode=True, seed=seed)
    value_function = value_function_class(env, initial_value, gamma, aggr_groups)
    max_value = np.max(np.abs(value_function.true_value_function))
    initial_rmse = value_function.calculate_rmse()
    results = []
    visits = np.ones(value_function.shape)
    rmse = 'rmse_' in to_return
    length = '_length' in to_return
    avg = 'avg_' in to_return
    ep = '_ep' in to_return
    step = '_step' in to_return
    incremental_mean = alpha_strategy == 'incremental_mean'
    overflow = False
    for episode in range(episodes):
        if not overflow:
            alpha_ep = alpha/np.power((episode+1), 1.0) if alpha_strategy=='inverse_ep' else alpha
            # Run episode
            env.reset()
            while not env.done:
                state, reward = env.step()
                if step:
                    results.append(value_function.calculate_rmse())
            # Learn values
            T = len(env.episode)-1
            gamma_powers = np.power(gamma, np.arange(0, T))
            lambda_powers = np.power(_lambda, np.arange(0, T))
            for t in range(T):
                # Calculate λ-return
                G = value_function.calculate_λreturn(t, T, lambda_powers, gamma_powers)
                # Calculate delta
                delta = G - value_function.get_value(env.episode[t][1])
                # Update value function or save update
                state_index = value_function.get_index(env.episode[t][1])
                if single_update:
                    value_function.save_update_state(env.episode[t][1], alpha_ep/visits[state_index] * delta)
                else:
                    value_function.update_state(env.episode[t][1], alpha_ep/visits[state_index] * delta)
                if incremental_mean:
                    visits[state_index] += 1.0
                # Avoid overflow
                if abs(value_function.get_value_by_index(state_index))>max_value*10000:
                    overflow = True
                    break
            if single_update:
                value_function.update_saved()
            if step:
                results.append(value_function.calculate_rmse())
        if ep:
            if length:
                results.append(env.episode_length)
            else:
                results.append(value_function.calculate_rmse())
    if to_return == 'value_function':
        return value_function.w
    if to_return == 'last_rmse_ep':
        return results[-1]
    if avg:
        return np.array(results).mean() 
    if rmse:
        results.insert(0, initial_rmse)
    return results

def λreturn_rec(_lambda, alpha, episodes, environment_class, env_n, gamma, value_function_class, initial_value, aggr_groups, alpha_strategy, single_update, to_return, seed):
    """
    λ-return algorithm with recursive calculation of λ-returns

    args:
        episodes: number of episodes
        environment_class: class of the environment
        env_n: size or name of the environment
        _lambda: value of lambda
        alpha: initial step size
        gamma: discount factor
        value_function_class: class of the state value function
        initial_value: initial value of the value function
        aggr_groups: number of groups for state aggregation
        alpha_strategy: strategy for the step size - fixed, inverse_ep, incremental_mean
        single_update: accumulate the updates and do a single update at the end of the episode
        to_return: object to return - value_function, avg_rmse_ep, rmse_ep, rmse_step, ep_length, avg_ep_length
        seed: numpy seed for reproducibility
    """
    env = environment_class(env_n, save_current_episode=True, seed=seed)
    value_function = value_function_class(env, initial_value, gamma, aggr_groups)
    max_value = np.max(np.abs(value_function.true_value_function))
    initial_rmse = value_function.calculate_rmse()
    results = []
    visits = np.ones(value_function.shape)
    rmse = 'rmse_' in to_return
    length = '_length' in to_return
    avg = 'avg_' in to_return
    ep = '_ep' in to_return
    step = '_step' in to_return
    incremental_mean = alpha_strategy == 'incremental_mean'
    overflow = False
    for episode in range(episodes):
        if not overflow:
            alpha_ep = alpha/np.power((episode+1), 1.0) if alpha_strategy=='inverse_ep' else alpha
            # Run episode
            env.reset()
            G = 0.0
            old_v = 0.0
            lambda_gamma_power = 1.0
            while not env.done:
                state, reward = env.step()
                new_v = 0.0 if env.done else value_function.get_value(state)
                G += lambda_gamma_power*(reward+gamma*new_v-old_v)
                lambda_gamma_power *= _lambda*gamma
                old_v = new_v
                if step:
                    results.append(value_function.calculate_rmse())
            # Learn values
            T = len(env.episode)-1
            gamma_powers = np.power(gamma, np.arange(0, T))
            lambda_powers = np.power(_lambda, np.arange(0, T))
            old_v = 0.0
            for t in range(T):
                # Calculate λ-return
                if t>0:
                    G = (G-env.episode[t][0])/gamma
                    d = 0.0
                    for n in range(1,T-t+1):
                        v_changed = value_function.states_are_equal(env.episode[t-1][1], env.episode[t+n-1][1])
                        delta_p = 0.0
                        if n == T-t:
                            delta_p = (env.episode[t+n][0] - (old_v if v_changed else value_function.get_value(env.episode[t+n-1][1])))
                        else:
                            delta_p = (env.episode[t+n][0] + gamma*value_function.get_value(env.episode[t+n][1]) - (old_v if v_changed else value_function.get_value(env.episode[t+n-1][1])))
                        d += gamma_powers[n-1]*lambda_powers[n-1]*delta_p
                    G += (1-_lambda)*d
                old_v = value_function.get_value(env.episode[t][1])
                # Calculate delta
                delta = G - value_function.get_value(env.episode[t][1])
                # Update value function or save update
                state_index = value_function.get_index(env.episode[t][1])
                if single_update:
                    value_function.save_update_state(env.episode[t][1], alpha_ep/visits[state_index] * delta)
                else:
                    value_function.update_state(env.episode[t][1], alpha_ep/visits[state_index] * delta)
                if incremental_mean:
                    visits[state_index] += 1.0
                # Avoid overflow
                if abs(value_function.get_value_by_index(state_index))>max_value*10000:
                    overflow = True
                    break
            if single_update:
                value_function.update_saved()
            if step:
                results.append(value_function.calculate_rmse())
        if ep:
            if length:
                results.append(env.episode_length)
            else:
                results.append(value_function.calculate_rmse())
    if to_return == 'value_function':
        return value_function.w
    if to_return == 'last_rmse_ep':
        return results[-1]
    if avg:
        return np.array(results).mean()  
    if rmse:
        results.insert(0, initial_rmse)
    return np.array(results)

def TTDλ_wrong(_lambda, alpha, episodes, environment_class, env_n, eta, gamma, value_function_class, initial_value, aggr_groups, alpha_strategy, single_update, to_return, seed):
    """
    TTD(λ) with implementation error (accumulation of round-off errors)

    args:
        episodes: number of episodes
        environment_class: class of the environment
        env_n: size or name of the environment 
        eta: truncation parameter. n is chosen such that (_lambda*gamma)^n < eta < (_lambda*gamma)^(n-1)
        _lambda: value of lambda
        alpha: initial step size
        gamma: discount factor
        value_function_class: class of the state value function
        initial_value: initial value of the value function
        aggr_groups: number of groups for state aggregation
        alpha_strategy: strategy for the step size - fixed, inverse_ep, incremental_mean
        single_update: accumulate the updates and do a single update at the end of the episode
        to_return: object to return - value_function, avg_rmse_ep, rmse_ep, rmse_step, last_rmse_ep, ep_length, avg_ep_length
        seed: numpy seed for reproducibility
    """
    
    env = environment_class(env_n, save_current_episode=True, seed=seed)
    value_function = value_function_class(env, initial_value, gamma, aggr_groups)
    max_value = np.max(np.abs(value_function.true_value_function))
    initial_rmse = value_function.calculate_rmse()
    results = []
    n_value = 1
    if _lambda*gamma > 0:
        n_value = ceil(log(eta)/log(gamma*_lambda)) if _lambda*gamma < 1 else np.inf
    visits = np.ones(value_function.shape)
    rmse = 'rmse_' in to_return
    length = '_length' in to_return
    avg = 'avg_' in to_return
    ep = 'ep' in to_return
    step = '_step' in to_return
    incremental_mean = alpha_strategy == 'incremental_mean'
    overflow = False
    for episode in range(episodes):
        if not overflow:
            n = n_value
            alpha_ep = alpha/(episode+1) if alpha_strategy=='inverse_ep' else alpha
            # Run episode
            state = env.reset()
            T = np.inf
            t = 0
            rho = []
            old_v = 0.0
            G = 0.0
            lambda_gamma_power = 1.0
            while t+1-n<=T-1:
                delta_p = 0.0
                if not env.done:
                    # Take an action
                    state, reward = env.step()
                    new_v = 0.0 if env.done else value_function.get_value(state)
                    rho.append(reward+gamma*(1-_lambda)*new_v)
                    if env.done:
                        T = t+1
                        n = min(n,T)# with n greater than T we would have same returns (it handles the case n=inf and slightly speeds up the algorithm) 
                    delta_p = reward+gamma*new_v-old_v
                    old_v = new_v
                if T-t>=1:
                    G += lambda_gamma_power*delta_p
                    if t<n-1:
                        lambda_gamma_power *= _lambda*gamma
                tau = t+1-n
                if 0<=tau<=T-1:
                    state_index = value_function.get_index(env.episode[tau][1])
                    delta = G - value_function.get_value(env.episode[tau][1])
                    if single_update:
                        value_function.save_update_state(env.episode[tau][1], alpha_ep/visits[state_index] * delta)
                    else:
                        value_function.update_state(env.episode[tau][1], alpha_ep/visits[state_index] * delta)
                    if incremental_mean:
                        visits[state_index] += 1.0
                    G = (G-rho[tau])/(_lambda*gamma)
                    # Avoid overflow
                    if abs(value_function.get_value(env.episode[tau][1]))>max_value*10000:
                        overflow = True
                        break
                if step:
                    results.append(value_function.calculate_rmse())
                t += 1
            if single_update:
                value_function.update_saved()
        if ep:
            if length:
                results.append(env.episode_length)
            else:
                results.append(value_function.calculate_rmse())
    if to_return == 'value_function':
        return value_function.w
    if to_return == 'last_rmse_ep':
        return results[-1]
    if avg:
        return np.array(results).mean()  
    if rmse:
        results.insert(0, initial_rmse)
    return np.array(results)

def TTDλ(_lambda, alpha, episodes, environment_class, env_n, eta, gamma, value_function_class, initial_value, aggr_groups, alpha_strategy, single_update, to_return, seed):
    """
    TTD(λ)

    args:
        episodes: number of episodes
        environment_class: class of the environment
        env_n: size or name of the environment 
        eta: truncation parameter. n chosen such that (_lambda*gamma)^n < eta < (_lambda*gamma)^(n-1)
        _lambda: value of lambda
        alpha: initial step size
        gamma: discount factor
        value_function_class: class of the state value function
        initial_value: initial value of the value function
        aggr_groups: number of groups for state aggregation
        alpha_strategy: strategy for the step size - fixed, inverse_ep, incremental_mean
        single_update: accumulate the updates and do a single update at the end of the episode
        to_return: object to return - value_function, avg_rmse_ep, rmse_ep, rmse_step, last_rmse_ep, ep_length, avg_ep_length
        seed: numpy seed for reproducibility
    """
    env = environment_class(env_n, save_current_episode=True, seed=seed)
    value_function = value_function_class(env, initial_value, gamma, aggr_groups)
    max_value = np.max(np.abs(value_function.true_value_function))
    initial_rmse = value_function.calculate_rmse()
    results = []
    n_value = 1
    if _lambda*gamma > 0:
        n_value = ceil(log(eta)/log(gamma*_lambda)) if _lambda*gamma < 1 else np.inf
    lambda_gamma_last_power = np.power(gamma*_lambda, n_value-1) if _lambda < 1 else None
    visits = np.ones(value_function.shape)
    rmse = 'rmse_' in to_return
    length = '_length' in to_return
    avg = 'avg_' in to_return
    ep = 'ep' in to_return
    step = '_step' in to_return
    incremental_mean = alpha_strategy == 'incremental_mean'
    overflow = False
    for episode in range(episodes):
        if not overflow:
            n = n_value
            alpha_ep = alpha/(episode+1) if alpha_strategy=='inverse_ep' else alpha
            # Run episode
            state = env.reset()
            T = np.inf
            t = 0
            rho = []
            old_v = 0.0
            G_calc = 0.0
            G = None
            i = 0
            lambda_gamma_power = 1.0
            while t+1-n<=T-1:
                if not env.done:
                    # Take an action
                    state, reward = env.step()
                    new_v = 0.0 if env.done else value_function.get_value(state)
                    rho.append(reward+gamma*(1-_lambda)*new_v)
                    delta_p = reward+gamma*new_v-old_v
                    old_v = new_v
                    G_calc += lambda_gamma_power*delta_p
                    if i==n-1:# truncated λ-return calculated -> start calculating a new return
                        G = G_calc
                        G_calc = old_v
                        lambda_gamma_power = 1.0
                        i = 0
                    else:
                        i+=1
                        lambda_gamma_power *= _lambda*gamma
                    if env.done:
                        T = t+1
                        if i>0:
                            G = G_calc if G is None else G+lambda_gamma_last_power*delta_p
                        n = min(n,T)# with n greater than T we would have same returns (it handles the case n=inf and slightly speeds up the algorithm)
                    elif G is not None and i>0: 
                        G += lambda_gamma_last_power*delta_p
                tau = t+1-n
                if 0<=tau<=T-1:
                    state_index = value_function.get_index(env.episode[tau][1])
                    delta = G - value_function.get_value(env.episode[tau][1])
                    if single_update:
                        value_function.save_update_state(env.episode[tau][1], alpha_ep/visits[state_index] * delta)
                    else:
                        value_function.update_state(env.episode[tau][1], alpha_ep/visits[state_index] * delta)
                    if incremental_mean:
                        visits[state_index] += 1.0
                    if n>1:
                        G = (G-rho[tau])/(_lambda*gamma)
                    # Avoid overflow
                    if abs(value_function.get_value_by_index(state_index))>max_value*10000:
                        overflow = True
                        break
                if step:
                    results.append(value_function.calculate_rmse())
                t += 1
            if single_update:
                value_function.update_saved()
        if ep:
            if length:
                results.append(env.episode_length)
            else:
                results.append(value_function.calculate_rmse())
    if to_return == 'value_function':
        return value_function.w
    if to_return == 'last_rmse_ep':
        return results[-1]
    if avg:
        return np.array(results).mean()  
    if rmse:
        results.insert(0, initial_rmse)
    return np.array(results)

def TDλ(_lambda, alpha, episodes, environment_class, env_n, algorithm_type, gamma, value_function_class, initial_value, aggr_groups, alpha_strategy, single_update, to_return, seed):
    """
    TD(λ)

    args:
        episodes: number of episodes
        environment_class: class of the environment
        env_n: size or name of the environment
        _lambda: value of lambda
        algorithm_type: type of algorithm (accumulate, replace, true_online)
        alpha: initial step size
        gamma: discount factor
        value_function_class: class of the state value function
        initial_value: initial value of the value function
        aggr_groups: number of groups for state aggregation
        alpha_strategy: strategy for the step size - fixed, inverse_ep, incremental_mean
        single_update: accumulate the updates and do a single update at the end of the episode
        to_return: object to return - value_function, avg_rmse_ep, rmse_ep, rmse_step, ep_length, avg_ep_length
        seed: numpy seed for reproducibility
    """
    if algorithm_type not in ['accumulate', 'replace', 'true_online']:
        print("Error: invalid algorithm_type. Values allowed: accumulate, replace and true_online")
        return
    env = environment_class(env_n, save_current_episode=False, seed=seed)
    value_function = value_function_class(env, initial_value, gamma, aggr_groups)
    max_value = np.max(np.abs(value_function.true_value_function))
    initial_rmse = value_function.calculate_rmse()
    results = []
    visits = np.ones(value_function.shape)
    rmse = 'rmse_' in to_return
    length = '_length' in to_return
    avg = 'avg_' in to_return
    ep = '_ep' in to_return
    step = '_step' in to_return
    incremental_mean = alpha_strategy == 'incremental_mean'
    overflow = False
    for episode in range(episodes):
        if not overflow:
            alpha_ep = alpha/(episode+1) if alpha_strategy=='inverse_ep' else alpha
            z = np.zeros(value_function.shape) # eligibility trace
            # Run episode
            t = 0
            state = env.reset()
            old_v = 0.0
            while not env.done and not overflow:
                # Take an action
                new_state, reward = env.step()
                # Update the trace vector
                state_index = value_function.get_index(state)
                _alpha = alpha_ep/visits[state_index]
                z = gamma*_lambda*z
                if algorithm_type == 'accumulate':
                    z[state_index] += 1.0
                elif algorithm_type == 'replace':
                    z[state_index] = 1.0
                else:
                    z[state_index] = z[state_index] + 1 - _alpha*z[state_index]
                # Calculate delta
                v = value_function.get_value(state)
                new_v = 0.0 if env.done else value_function.get_value(new_state)
                delta = reward + gamma*new_v - v
                # Update values
                if single_update:
                    value_function.save_update(_alpha*(delta + (v-old_v if algorithm_type == 'true_online' else 0.0))*z)
                    value_function.save_update_state(state, -_alpha*(v-old_v) if algorithm_type == 'true_online' else 0.0)
                else:
                    value_function.update(_alpha*(delta + (v-old_v if algorithm_type == 'true_online' else 0.0))*z)
                    value_function.update_state(state, -_alpha*(v-old_v) if algorithm_type == 'true_online' else 0.0)
                if incremental_mean:
                    visits[state_index] += 1.0
                # Avoid overflow
                if np.max(np.abs(value_function.w))>max_value*10000:
                    overflow = True
                    break
                # Update variables for the next step
                old_v = new_v
                state = new_state
                if step:
                    results.append(value_function.calculate_rmse())
                t+=1
            if single_update:
                value_function.update_saved()
        if ep:
            if length:
                results.append(env.episode_length)
            else:
                results.append(value_function.calculate_rmse())
    if to_return == 'value_function':
        return value_function.w
    if to_return == 'last_rmse_ep':
        return results[-1]
    if avg:
        return np.array(results).mean()  
    if rmse:
        results.insert(0, initial_rmse)
    return np.array(results)

def ETλ(_lambda, alpha, episodes, environment_class, env_n, algorithm_type, beta, gamma, value_function_class, initial_value, aggr_groups, alpha_strategy, single_update, to_return, seed):
    """
    ET(λ) algorithm

    args:
        episodes: number of episodes
        environment_class: class of the environment
        env_n: size of name of the environment
        _lambda: value of lambda
        algorithm_type: type of algorithm (accumulate, replace, true_online)
        alpha: initial step size
        gamma: discount factor
        value_function_class: class of the state value function
        initial_value: initial value of the value function
        aggr_groups: number of groups for state aggregation
        alpha_strategy: strategy for the step size - fixed, inverse_ep, incremental_mean
        single_update: accumulate the updates and do a single update at the end of the episode
        to_return: object to return - value_function, avg_rmse_ep, rmse_ep, rmse_step, ep_length, avg_ep_length
        seed: numpy seed for reproducibility
    """
    if algorithm_type not in ['accumulate', 'replace', 'dutch', 'true_online']:
        print("Error: invalid algorithm_type. Values allowed: accumulate, replace and true_online")
        return
    env = environment_class(env_n, save_current_episode=False, seed=seed)
    value_function = value_function_class(env, initial_value, gamma, aggr_groups)
    max_value = np.max(np.abs(value_function.true_value_function))
    initial_rmse = value_function.calculate_rmse()
    results = []
    visits = np.ones(value_function.shape)
    visits_trace = np.zeros(value_function.shape)
    rmse = 'rmse_' in to_return
    length = '_length' in to_return
    avg = 'avg_' in to_return
    ep = '_ep' in to_return
    step = '_step' in to_return
    incremental_mean = alpha_strategy == 'incremental_mean'
    e = np.zeros((*value_function.shape, *value_function.shape)) # expected trace
    overflow = False
    for episode in range(episodes):
        if not overflow:
            alpha_ep = alpha/(episode+1) if alpha_strategy=='inverse_ep' else alpha
            z = np.zeros(value_function.shape) # eligibility trace
            # Run episode
            t = 0
            state = env.reset()
            old_v = 0.0
            while not env.done:
                # Take an action
                new_state, reward = env.step()
                # Update the trace vector
                state_index = value_function.get_index(state if isinstance(state, int) else tuple(state))
                _alpha = alpha_ep/visits[state_index]
                z = gamma*_lambda*z
                if algorithm_type == 'accumulate':
                    z[state_index] += 1.0
                elif algorithm_type == 'replace':
                    z[state_index] = 1.0
                else:
                    z[state_index] = z[state_index] + 1 - _alpha*z[state_index]
                visits_trace[state_index] += 1.0
                # e[state_index] += 1.0/visits_trace[state_index]*(z-e[state_index]) if beta is None else beta*(z-e[state_index])
                e[state_index] += 1.0/visits_trace[state_index]*(z-e[state_index]) if beta is None else beta*(z-e[state_index])
                # Calculate delta
                v = value_function.get_value(state)
                new_v = 0.0 if env.done else value_function.get_value(new_state)
                delta = reward + gamma*new_v - v
                # Update values            
                if single_update:
                    value_function.save_update(_alpha*(delta + (v-old_v if algorithm_type == 'true_online' else 0.0))*e[state_index])
                    value_function.save_update_state(state, -_alpha*(v-old_v) if algorithm_type == 'true_online' else 0.0)
                else:
                    value_function.update(_alpha*(delta + (v-old_v if algorithm_type == 'true_online' else 0.0))*e[state_index])
                    value_function.update_state(state, -_alpha*(v-old_v) if algorithm_type == 'true_online' else 0.0)
                if incremental_mean:
                    visits[state_index] += 1.0
                # Avoid overflow
                if np.max(np.abs(value_function.w))>max_value*1000:
                    overflow = True
                    break
                # Update variables for the next step
                old_v = new_v
                state = new_state
                if step:
                    results.append(value_function.calculate_rmse())
                t+=1
            if single_update:
                value_function.update_saved()
        if ep:
            if length:
                results.append(env.episode_length)
            else:
                results.append(value_function.calculate_rmse())
    if to_return == 'value_function':
        return value_function.w
    if to_return == 'last_rmse_ep':
        return results[-1]
    if avg:
        return np.array(results).mean(axis=0)  
    if rmse:
        results.insert(0, initial_rmse)
    return np.array(results)

def nstepSARSA(n, alpha, episodes, environment_class, epsilon, env_n, gamma, value_function_class, initial_value, aggr_groups, alpha_strategy, max_steps, single_update, to_return, seed):
    """
    n-step SARSA algorithm

    args:
        episodes: number of episodes
        environment_class: class of the environment
        env_n: size or name of the environment
        n: value of n
        alpha: initial step size
        gamma: discount factor
        epsilon: 
        value_function_class: class of the action value function
        initial_value: initial value of the value function
        aggr_groups: number of groups for state aggregation or number of tilings
        alpha_strategy: strategy for the step size - fixed, inverse_ep
        max_steps: steps before episode truncation
        single_update: accumulate the updates and do a single update at the end of the episode
        to_return: object to return - action_value_function, ep_length, ep_return, avg_ep_return
        seed: numpy seed for reproducibility
    """
    results = []
    env = environment_class(env_n, save_current_episode=True, seed=seed)
    action_value_function = value_function_class(env, initial_value, aggr_groups)
    gamma_powers = np.power(gamma, np.arange(0, n+1))
    avg = 'avg_' in to_return
    ep = '_ep' in to_return
    step = '_step' in to_return
    incremental_mean = alpha_strategy == 'incremental_mean'
    overflow = False
    for episode in range(episodes):
        if not overflow:
            n_ep = n
            alpha_ep = alpha/(episode+1) if alpha_strategy=='inverse_ep' else alpha
            # Run episode
            state = env.reset()
            action = action_value_function.choose_action(epsilon)
            T = np.inf
            t = 0
            while t+1-n_ep<=T-1 and t<max_steps:            
                if not env.done:
                    # epsilon = epsilon+0.01*(0.001-epsilon)
                    # Take an action
                    state, reward = env.step(action)
                    # If not done, choose the next action
                    if env.done:
                        T = t+1
                        n_ep = min(n_ep, T)
                    else:
                        action = action_value_function.choose_action(epsilon)
                tau = t+1-n_ep
                if 0<=tau<=T-1:
                    r, s, a = env.episode[tau]
                    G = action_value_function.calculate_return(tau, n_ep, T, gamma_powers)
                    delta = G - action_value_function.get_value(s,a)
                    index = action_value_function.get_index(s,a)
                    if single_update:
                        action_value_function.save_update_state(s, a, alpha_ep * delta)
                    else:
                        action_value_function.update_state(s, a, alpha_ep * delta)
                    # Avoid overflow
                    if abs(action_value_function.get_value_by_index(index))>100*max_steps:
                        overflow = True
                        env.episode_return = -float(max_steps)
                        env.episode_length = max_steps
                        break
                t += 1
            if single_update:
                action_value_function.update_saved()
            results.append(env.episode_length if to_return=='ep_length' else env.episode_return)
        else:
            results.append(env.episode_length if to_return=='ep_length' else -float(max_steps))
    if avg:
        return np.array(results).mean()
    return results if to_return != 'action_value_function' else action_value_function

def nstepSARSA_rec(n, alpha, episodes, environment_class, epsilon, env_n, gamma, value_function_class, initial_value, aggr_groups, alpha_strategy, max_steps, single_update, to_return, seed):
    """
    n-step SARSA algorithm

    args:
        episodes: number of episodes
        environment_class: class of the environment
        env_n: size or name of the environment
        n: value of n
        alpha: initial step size
        gamma: discount factor
        epsilon: 
        value_function_class: class of the action value function
        initial_value: initial value of the value function
        aggr_groups: number of groups for state aggregation or number of tilings
        alpha_strategy: strategy for the step size - fixed, inverse_ep
        max_steps:
        single_update: accumulate the updates and do a single update at the end of the episode
        to_return: object to return - action_value_function, ep_length, ep_return, avg_ep_return
        seed: numpy seed for reproducibility
    """
    results = []
    env = environment_class(env_n, save_current_episode=True, seed=seed)
    action_value_function = value_function_class(env, initial_value, aggr_groups)
    gamma_last_power = np.power(gamma, n-1)
    avg = 'avg_' in to_return
    ep = '_ep' in to_return
    step = '_step' in to_return
    incremental_mean = alpha_strategy == 'incremental_mean'
    overflow = False
    for episode in range(episodes):
        if not overflow:
            n_ep = n
            alpha_ep = alpha/(episode+1) if alpha_strategy=='inverse_ep' else alpha
            # Run episode
            state = env.reset()
            action = action_value_function.choose_action(epsilon)
            T = np.inf
            t = 0
            old_q = 0.0
            G_calc = 0.0
            G = None
            i = 0
            gamma_power = 1.0
            while t+1-n_ep<=T-1 and t<max_steps:
                if not env.done:
                    # epsilon = epsilon+0.01*(0.001-epsilon)
                    # Take an action
                    state, reward = env.step(action)
                    if not env.done:
                        action = action_value_function.choose_action(epsilon)
                    new_q = 0.0 if env.done else action_value_function.get_value(state,action)
                    delta_p = reward+gamma*new_q-old_q
                    old_q = new_q
                    G_calc += gamma_power*delta_p
                    if i==n_ep-1:# n-step return calculated -> start calculating a new return
                        G = G_calc
                        G_calc = old_q
                        gamma_power = 1.0
                        i = 0
                    else:
                        i+=1
                        gamma_power *= gamma
                    if env.done:
                        T = t+1
                        if i>0:
                            G = G_calc if G is None else G+gamma_last_power*delta_p
                        n_ep = min(n_ep,T)# with n greater than T we would have same returns (slightly speeds up the algorithm)
                    elif G is not None and i>0: 
                        G += gamma_last_power*delta_p
                tau = t+1-n_ep
                if 0<=tau<=T-1:
                    r, s, a = env.episode[tau]
                    delta = G - action_value_function.get_value(s,a)
                    index = action_value_function.get_index(s,a)
                    if single_update:
                        action_value_function.save_update_state(s, a, alpha_ep * delta)
                    else:
                        action_value_function.update_state(s, a, alpha_ep * delta)
                    if n_ep>1:
                        G = (G-env.episode[tau+1][0])/gamma    
                    # Avoid overflow
                    if abs(action_value_function.get_value_by_index(index))>100*max_steps:
                        overflow = True
                        env.episode_return = -float(max_steps)
                        env.episode_length = max_steps
                        break
                t += 1
            if single_update:
                action_value_function.update_saved()
            results.append(env.episode_length if to_return=='ep_length' else env.episode_return)
        else:
            results.append(env.episode_length if to_return=='ep_length' else -float(max_steps))
    if avg:
        return np.array(results).mean()
    return results if to_return != 'action_value_function' else action_value_function

def SARSAλ(_lambda, alpha, episodes, environment_class, epsilon, env_n, algorithm_type, gamma, value_function_class, initial_value, aggr_groups, alpha_strategy, max_steps, single_update, to_return, seed):
    """
    SARSA(λ) algorithm

    args:
        episodes: number of episodes
        environment_class: class of the environment
        env_n: size or name of the environment
        _lambda: value of lambda
        algorithm_type: type of algorithm (accumulate, replace, true_online)
        alpha: initial step size
        gamma: discount factor
        value_function_class: class of the action value function
        initial_value: initial value of the value function
        aggr_groups: number of groups for state aggregation or number of tilings
        alpha_strategy: strategy for the step size - fixed, inverse_ep
        single_update: accumulate the updates and do a single update at the end of the episode
        to_return: object to return - action_value_function, ep_length, ep_return, avg_ep_return
        seed: numpy seed for reproducibility
    """
    if algorithm_type not in ['accumulate', 'replace', 'replace_and_clear', 'true_online']:
        print("Error: invalid algorithm_type. Values allowed: accumulate, replace, replace_and_clear and true_online")
        return
    results = []
    env = environment_class(env_n, save_current_episode=True, seed=seed)
    action_value_function = value_function_class(env, initial_value, aggr_groups)
    avg = 'avg_' in to_return
    ep = '_ep' in to_return
    step = '_step' in to_return
    overflow = False
    for episode in range(episodes):
        if not overflow:
            alpha_ep = alpha/(episode+1) if alpha_strategy=='inverse_ep' else alpha
            z = np.zeros_like(action_value_function.w) # trace
            # Run episode
            state = env.reset()
            action = action_value_function.choose_action(epsilon)
            old_q = 0.0
            index = action_value_function.get_index(state, action)
            while not env.done and env.episode_length<max_steps:
                # Take an action
                new_state, reward = env.step(action)
                # Update the trace vector
                z = gamma*_lambda*z
                if algorithm_type == 'accumulate':
                    z[index] += 1.0
                elif algorithm_type in ['replace','replace_and_clear']:
                    z[index] = 1.0
                else:
                    z[index] = z[index] + 1 - alpha*np.sum(z[index])
                if 'clear' in algorithm_type:
                    for a in range(env.num_actions):
                        if a != action:
                            z[action_value_function.get_index(state, a)] = 0.0
                # Choose the next action and calculate the value of new_state,new_action
                new_action = action_value_function.choose_action(epsilon)
                new_index = action_value_function.get_index(new_state, new_action)
                new_q = action_value_function.get_value_by_index(new_index)
                # Calculate delta
                q = action_value_function.get_value_by_index(index)
                delta = reward + gamma*new_q - q
                # Update weights vector
                if single_update:
                    action_value_function.save_update(alpha_ep*(delta + (q-old_q if algorithm_type == 'true_online' else 0.0))*z)
                    action_value_function.save_update_state(state, action, -alpha_ep*(q-old_q) if algorithm_type == 'true_online' else 0.0)
                else:
                    action_value_function.update(alpha_ep*(delta + (q-old_q if algorithm_type == 'true_online' else 0.0))*z)
                    action_value_function.update_state(state, action, -alpha_ep*(q-old_q) if algorithm_type == 'true_online' else 0.0)
                # Avoid overflow
                if np.max(np.abs(action_value_function.w))>100*max_steps:
                    overflow = True
                    env.episode_return = -float(max_steps)
                    env.episode_length = max_steps
                    break
                # Update variables for the next step
                old_q = new_q
                state = new_state
                action = new_action
                index = new_index
            if single_update:
                action_value_function.update_saved()
            results.append(env.episode_length if to_return=='ep_length' else env.episode_return)
        else:
            results.append(env.episode_length if to_return=='ep_length' else -float(max_steps))
    if avg:
        return np.array(results).mean(axis=0)
    return results if to_return != 'action_value_function' else action_value_function

def SARSAλ_tiling(_lambda, alpha, episodes, environment_class, epsilon, env_n, algorithm_type, gamma, value_function_class, initial_value, aggr_groups, alpha_strategy, max_steps, single_update, to_return, seed):
    """
    SARSA(λ) algorithm (with optimization for action value function with tile coding)

    args:
        episodes: number of episodes
        environment_class: class of the environment
        env_n: size or name of the environment
        _lambda: value of lambda
        algorithm_type: type of algorithm (accumulate, replace, true_online)
        alpha: initial step size
        gamma: discount factor
        value_function_class: class of the action value function
        initial_value: initial value of the value function
        aggr_groups: number of groups for state aggregation or number of tilings
        alpha_strategy: strategy for the step size - fixed, inverse_ep
        single_update: accumulate the updates and do a single update at the end of the episode
        to_return: object to return - action_value_function, ep_length, ep_return, avg_ep_return
        seed: numpy seed for reproducibility
    """
    if algorithm_type not in ['accumulate', 'replace', 'replace_and_clear', 'true_online']:
        print("Error: invalid algorithm_type. Values allowed: accumulate, replace, replace_and_clear and true_online")
        return
    results = []
    env = environment_class(env_n, save_current_episode=False, seed=seed)
    action_value_function = value_function_class(env, initial_value, aggr_groups)
    avg = 'avg_' in to_return
    ep = '_ep' in to_return
    step = '_step' in to_return
    overflow = False
    for episode in range(episodes):
        if not overflow:
            alpha_ep = alpha/(episode+1) if alpha_strategy=='inverse_ep' else alpha
            # Run episode
            state = env.reset()
            action = action_value_function.choose_action(epsilon)
            old_q = 0.0
            index = action_value_function.get_index(state, action)
            z = np.zeros_like(action_value_function.w) # trace
            is_active = []# list of active indexes to speed up the algorithm
            while not env.done and env.episode_length<max_steps:
                # Take an action
                new_state, reward = env.step(action)
                # Update the trace vector
                z[is_active] = gamma*_lambda*z[is_active]
                if algorithm_type == 'accumulate':
                    z[index] += 1.0
                elif algorithm_type in ['replace','replace_and_clear']:
                    z[index] = 1.0
                else:
                    z[index] = z[index] + 1 - alpha*np.sum(z[index])
                if 'clear' in algorithm_type:
                    for a in range(env.num_actions):
                        if a != action:
                            z[action_value_function.get_index(state, a)] = 0.0
                # adding indexes to the list of active indexes
                for i in index:
                    if i not in is_active:
                        is_active.append(i)
                # Choose the next action and calculate the value of new_state,new_action
                new_action = action_value_function.choose_action(epsilon)
                new_index = action_value_function.get_index(new_state, new_action)
                # pad trace if needed
                if len(new_index) > 0:
                    max_index = max(new_index)
                    if max_index>z.size-1:
                        z = np.pad(z, (0,max_index+1-z.size))
                new_q = action_value_function.get_value_by_index(new_index)
                # Calculate delta
                q = action_value_function.get_value_by_index(index)
                delta = reward + gamma*new_q - q
                # Update weights vector
                if single_update:
                    action_value_function.save_update_by_index(is_active, alpha_ep*(delta + (q-old_q if algorithm_type == 'true_online' else 0.0))*z[is_active])
                    action_value_function.save_update_state(state, action, -alpha_ep*(q-old_q) if algorithm_type == 'true_online' else 0.0)
                else:
                    action_value_function.update_by_index(is_active, alpha_ep*(delta + (q-old_q if algorithm_type == 'true_online' else 0.0))*z[is_active])
                    action_value_function.update_state(state, action, -alpha_ep*(q-old_q) if algorithm_type == 'true_online' else 0.0)
                # Avoid overflow
                if np.max(np.abs(action_value_function.w[is_active]))>100*max_steps:
                    overflow = True
                    env.episode_return = -float(max_steps)
                    env.episode_length = max_steps
                    break
                # Update variables for the next step
                old_q = new_q
                state = new_state
                action = new_action
                index = new_index
            if single_update:
                action_value_function.update_saved()
            results.append(env.episode_length if to_return=='ep_length' else env.episode_return)
        else:
            results.append(env.episode_length if to_return=='ep_length' else -float(max_steps))
    if avg:
        return np.array(results).mean(axis=0)
    return results if to_return != 'action_value_function' else action_value_function

def SARSA_ETλ(_lambda, alpha, episodes, environment_class, epsilon, env_n, algorithm_type, beta, gamma, value_function_class, initial_value, aggr_groups, alpha_strategy, max_steps, single_update, to_return, seed):
    """
    SARSA(λ) algorithm

    args:
        episodes: number of episodes
        environment_class: class of the environment
        env_n: size or name of the environment
        _lambda: value of lambda
        algorithm_type: type of algorithm (accumulate, replace, true_online)
        alpha: initial step size
        gamma: discount factor
        value_function_class: class of the action value function
        initial_value: initial value of the value function
        aggr_groups: number of groups for state aggregation or number of tilings
        alpha_strategy: strategy for the step size - fixed, inverse_ep
        single_update: accumulate the updates and do a single update at the end of the episode
        to_return: object to return - action_value_function, ep_length, ep_return, avg_ep_return
        seed: numpy seed for reproducibility
    """
    if algorithm_type not in ['accumulate', 'replace', 'replace_and_clear', 'true_online']:
        print("Error: invalid algorithm_type. Values allowed: accumulate, replace, replace_and_clear and true_online")
        return
    results = []
    env = environment_class(env_n, save_current_episode=True, seed=seed)
    action_value_function = value_function_class(env, initial_value, aggr_groups)
    visits_trace = np.zeros(action_value_function.shape)
    avg = 'avg_' in to_return
    ep = '_ep' in to_return
    step = '_step' in to_return
    e = np.zeros((*action_value_function.shape, *action_value_function.shape)) # expected trace
    overflow = False
    for episode in range(episodes):
        if not overflow:
            alpha_ep = alpha/(episode+1) if alpha_strategy=='inverse_ep' else alpha
            z = np.zeros_like(action_value_function.w) # trace
            # Run episode
            state = env.reset()
            action = action_value_function.choose_action(epsilon)
            old_q = 0.0
            index = action_value_function.get_index(state, action)
            while not env.done and env.episode_length<max_steps:
                # Take an action
                new_state, reward = env.step(action)
                # Update the trace vector
                z = gamma*_lambda*z
                if algorithm_type == 'accumulate':
                    z[index] += 1.0
                elif algorithm_type in ['replace','replace_and_clear']:
                    z[index] = 1.0
                else:
                    z[index] = z[index] + 1 - alpha*np.sum(z[index])
                if 'clear' in algorithm_type:
                    for a in range(env.num_actions):
                        if a != action:
                            z[action_value_function.get_index(state, a)] = 0.0
                visits_trace[index] += 1.0
                e[index] += 1.0/visits_trace[index]*(z-e[index])# if beta is None else beta*(z-e[index])
                # e[index] += beta*(z-e[index])
                # Choose the next action and calculate the value of new_state,new_action
                new_action = action_value_function.choose_action(epsilon)
                new_index = action_value_function.get_index(new_state, new_action)
                new_q = action_value_function.get_value_by_index(new_index)
                # Calculate delta
                q = action_value_function.get_value_by_index(index)
                delta = reward + gamma*new_q - q
                # Update weights vector
                if single_update:
                    action_value_function.save_update(alpha_ep*(delta + (q-old_q if algorithm_type == 'true_online' else 0.0))*e[index])
                    action_value_function.save_update_state(state, action, -alpha_ep*(q-old_q) if algorithm_type == 'true_online' else 0.0)
                else:
                    action_value_function.update(alpha_ep*(delta + (q-old_q if algorithm_type == 'true_online' else 0.0))*e[index])
                    action_value_function.update_state(state, action, -alpha_ep*(q-old_q) if algorithm_type == 'true_online' else 0.0)
                # Avoid overflow
                if np.max(np.abs(action_value_function.w))>100*max_steps:
                    overflow = True
                    env.episode_return = -float(max_steps)
                    env.episode_length = max_steps
                    break
                # Update variables for the next step
                old_q = new_q
                state = new_state
                action = new_action
                index = new_index
            if single_update:
                action_value_function.update_saved()
            results.append(env.episode_length if to_return=='ep_length' else env.episode_return)
        else:
            results.append(env.episode_length if to_return=='ep_length' else -float(max_steps))
    if avg:
        return np.array(results).mean(axis=0)
    return results if to_return != 'action_value_function' else action_value_function
