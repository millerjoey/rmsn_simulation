# Import 
import simulation.cancer_simulation as sim
import pickle
import numpy as np
import logging

def _generate_intervention_data(save = False, assigned_actions = None):
    chemo_coeff = 10
    radio_coeff = 10
    window_size = 15
    num_time_steps = 120
    t_intervention = 1
    np.random.seed(100)
    num_patients = 10000
    pickle_file = 'models/cancer_sim_intervention_{}_{}_t1final.pkl'.format(chemo_coeff, radio_coeff, t_intervention)

    params = sim.get_confounding_params(num_patients, chemo_coeff=chemo_coeff,
                                        radio_coeff=radio_coeff)
    params['window_size'] = window_size
    training_data = simulate_intervention_assigned(params, num_time_steps, t_intervention, assigned_actions)

    params = sim.get_confounding_params(int(num_patients / 10), chemo_coeff=chemo_coeff,
                                        radio_coeff=radio_coeff)
    params['window_size'] = window_size
    validation_data = simulate_intervention_assigned(params, num_time_steps, t_intervention, assigned_actions)

    params = sim.get_confounding_params(int(num_patients / 10), chemo_coeff=chemo_coeff,
                                        radio_coeff=radio_coeff)
    params['window_size'] = window_size
    test_data = simulate_intervention_assigned(params, num_time_steps, t_intervention, assigned_actions)

    scaling_data = sim.get_scaling_params(training_data)

    pickle_map = {'chemo_coeff': chemo_coeff,
                    'radio_coeff': radio_coeff,
                    'num_time_steps': num_time_steps,
                    'training_data': training_data,
                    'validation_data': validation_data,
                    'test_data': test_data,
                    'scaling_data': scaling_data,
                    'window_size': window_size}

    logging.info("Saving pickle map to {}".format(pickle_file))
    if save:
        pickle.dump(pickle_map, open(pickle_file, 'wb'))
    return pickle_map


#def simulate_intervention(simulation_params, num_time_steps, t_intervention, assigned_actions=None):
    """
    Core routine to generate simulation paths

    :param simulation_params:
    :param num_time_steps:
    :param assigned_actions:
    :return:
    """

    total_num_radio_treatments = 1
    total_num_chemo_treatments = 1

    radio_amt = np.array([2.0 for i in range(total_num_radio_treatments)])  # Gy
    radio_days = np.array([i + 1 for i in range(total_num_radio_treatments)])
    chemo_amt = [5.0 for i in range(total_num_chemo_treatments)]
    chemo_days = [(i + 1) * 7 for i in range(total_num_chemo_treatments)]

    # sort this
    chemo_idx = np.argsort(chemo_days)
    chemo_amt = np.array(chemo_amt)[chemo_idx]
    chemo_days = np.array(chemo_days)[chemo_idx]

    drug_half_life = 1  # one day half life for drugs

    # Unpack simulation parameters
    initial_stages = simulation_params['initial_stages']
    initial_volumes = simulation_params['initial_volumes']
    alphas = simulation_params['alpha']
    rhos = simulation_params['rho']
    betas = simulation_params['beta']
    beta_cs = simulation_params['beta_c']
    Ks = simulation_params['K']
    patient_types = simulation_params['patient_types']
    window_size = simulation_params['window_size']  # controls the lookback of the treatment assignment policy

    # Coefficients for treatment assignment probabilities
    chemo_sigmoid_intercepts = simulation_params['chemo_sigmoid_intercepts']
    radio_sigmoid_intercepts = simulation_params['radio_sigmoid_intercepts']
    chemo_sigmoid_betas = simulation_params['chemo_sigmoid_betas']
    radio_sigmoid_betas = simulation_params['radio_sigmoid_betas']

    num_patients = initial_stages.shape[0]

    # Commence Simulation
    cancer_volume = np.zeros((num_patients, num_time_steps))
    chemo_dosage = np.zeros((num_patients, num_time_steps))
    radio_dosage = np.zeros((num_patients, num_time_steps))
    chemo_application_point = np.zeros((num_patients, num_time_steps))
    radio_application_point = np.zeros((num_patients, num_time_steps))
    sequence_lengths = np.zeros(num_patients)
    death_flags = np.zeros((num_patients, num_time_steps))
    recovery_flags = np.zeros((num_patients, num_time_steps))
    chemo_probabilities = np.zeros((num_patients, num_time_steps))
    radio_probabilities = np.zeros((num_patients, num_time_steps))

    # Counterfactuals
    cancer_volume_counterfactual = np.zeros((num_patients, num_time_steps))
    chemo_dosage_counterfactual = np.zeros((num_patients, num_time_steps))
    radio_dosage_counterfactual = np.zeros((num_patients, num_time_steps))
    chemo_application_point_counterfactual = np.zeros((num_patients, num_time_steps))
    radio_application_point_counterfactual = np.zeros((num_patients, num_time_steps))

    noise_terms = 0.01 * np.random.randn(num_patients,
                                         num_time_steps)  # 5% cell variability
    recovery_rvs = np.random.rand(num_patients, num_time_steps)

    chemo_application_rvs = np.random.rand(num_patients, num_time_steps)
    radio_application_rvs = np.random.rand(num_patients, num_time_steps)

    # Run actual simulation
    for i in range(num_patients):

        logging.info("Simulating patient {} of {}".format(i + 1, num_patients))
        noise = noise_terms[i]

        # initial values
        cancer_volume[i, 0] = initial_volumes[i]
        alpha = alphas[i]
        beta = betas[i]
        beta_c = beta_cs[i]
        rho = rhos[i]
        K = Ks[i]


        # Setup cell volume
        b_death = False
        b_recover = False
        for t in range(1, num_time_steps):

            cancer_volume[i, t] = abs(cancer_volume[i, t - 1] * (1 + \
                                  + rho * np.log(K / cancer_volume[i, t - 1]) \
                                  - beta_c * chemo_dosage[i, t - 1] \
                                  - (alpha * radio_dosage[i, t - 1] + beta * radio_dosage[i, t - 1] ** 2) \
                                  + noise[t]))  # add noise to fit residuals
            
            if t >= t_intervention:
                if t == t_intervention: # Initialize whole row from cancer_volume. After t, we update using counterfactuals only.
                    cancer_volume_counterfactual[i, :] = cancer_volume[i, :]

                cancer_volume_counterfactual[i, t] = abs(cancer_volume_counterfactual[i, t - 1] * (1 + \
                                  + rho * np.log(K / cancer_volume_counterfactual[i, t - 1]) \
                                  - beta_c * chemo_dosage_counterfactual[i, t - 1] \
                                  - (alpha * radio_dosage_counterfactual[i, t - 1] + beta * radio_dosage_counterfactual[i, t - 1] ** 2) \
                                  + noise[t]))

            current_chemo_dose = 0.0
            previous_chemo_dose = 0.0 if t == 0 else chemo_dosage[i, t-1]

            # Action probabilities + death or recovery simulations
            cancer_volume_used = cancer_volume[i, max(t - window_size, 0):t]
            cancer_diameter_used = np.array([sim.calc_diameter(vol) for vol in cancer_volume_used]).mean()  # mean diameter over 15 days
            cancer_metric_used = cancer_diameter_used

            radio_prob = (1.0 / (1.0 + np.exp(-radio_sigmoid_betas[i]
                                            *(cancer_metric_used - radio_sigmoid_intercepts[i]))))
            chemo_prob = (1.0 / (1.0 + np.exp(- chemo_sigmoid_betas[i] *
                                                (cancer_metric_used - chemo_sigmoid_intercepts[i]))))

            chemo_probabilities[i, t] = chemo_prob
            radio_probabilities[i, t] = radio_prob

            # Action application
            if radio_application_rvs[i, t] < radio_prob :

                    radio_application_point[i, t] = 1
                    radio_dosage[i, t] = radio_amt[0]

            if chemo_application_rvs[i, t] < chemo_prob:

                # Apply chemo treatment
                chemo_application_point[i, t] = 1
                current_chemo_dose = chemo_amt[0]

            # Update chemo dosage
            chemo_dosage[i, t] = previous_chemo_dose * np.exp(-np.log(2) / drug_half_life) + current_chemo_dose

                        # Update all counterfactuals
            if t >= t_intervention and i < 5000:
                # Hold-out patients
                if t == t_intervention:
                    previous_chemo_dose_counterfactual = previous_chemo_dose
                else:
                    previous_chemo_dose_counterfactual = chemo_dosage_counterfactual[i, t-1]

                radio_dosage_counterfactual[i, t] = 0
                chemo_dosage_counterfactual[i, t] = previous_chemo_dose_counterfactual * np.exp(-np.log(2) / drug_half_life)

                # if t == t_intervention:
                #     previous_chemo_dose_counterfactual = previous_chemo_dose
                # else:
                #     previous_chemo_dose_counterfactual = chemo_dosage_counterfactual[i, t-1]
                
                # chemo_application_point_counterfactual[i, t] = assigned_actions[i, t, 0]
                # radio_application_point_counterfactual[i, t] = assigned_actions[i, t, 1]
                # current_chemo_dose = chemo_amt[0] if chemo_application_point_counterfactual[i, t] == 1 else 0
                # radio_dosage_counterfactual[i, t] = radio_amt[0] if radio_application_point_counterfactual[i, t] == 1 else 0
                # chemo_dosage_counterfactual[i, t] = previous_chemo_dose_counterfactual * np.exp(-np.log(2) / drug_half_life) + current_chemo_dose
            elif t >= t_intervention and i >= 5000:
                if t == t_intervention:
                    previous_chemo_dose_counterfactual = previous_chemo_dose
                else:
                    previous_chemo_dose_counterfactual = chemo_dosage_counterfactual[i, t-1]

                # Don't use assigned_actions. Instead, do the same as for t < t_intervention but double the probability of treatment:
                chemo_application_point_counterfactual[i, t] = 1 if chemo_application_rvs[i, t] < 2 * chemo_prob else 0
                radio_application_point_counterfactual[i, t] = 1 if radio_application_rvs[i, t] < 2 * radio_prob else 0
                current_chemo_dose = chemo_amt[0] if chemo_application_point_counterfactual[i, t] == 1 else 0
                radio_dosage_counterfactual[i, t] = radio_amt[0] if radio_application_point_counterfactual[i, t] == 1 else 0
                chemo_dosage_counterfactual[i, t] = previous_chemo_dose_counterfactual * np.exp(-np.log(2) / drug_half_life) + current_chemo_dose
            else:
                chemo_application_point_counterfactual[i, t] = chemo_application_point[i, t]
                radio_application_point_counterfactual[i, t] = radio_application_point[i, t]
                chemo_dosage_counterfactual[i, t] = chemo_dosage[i, t]
                radio_dosage_counterfactual[i, t] = radio_dosage[i, t]


        # Package outputs
        sequence_lengths[i] = int(t+1)
        death_flags[i, t] = 1 if b_death else 0
        recovery_flags[i, t] = 1 if b_recover else 0

    outputs = {'cancer_volume': cancer_volume,
               'cancer_volume_counterfactual': cancer_volume_counterfactual,
               'chemo_dosage_counterfactual': chemo_dosage_counterfactual,
               'radio_dosage_counterfactual': radio_dosage_counterfactual,
                'chemo_application_counterfactual': chemo_application_point_counterfactual,
                'radio_application_counterfactual': radio_application_point_counterfactual,
               'chemo_dosage': chemo_dosage,
               'radio_dosage': radio_dosage,
               'chemo_application': chemo_application_point,
               'radio_application': radio_application_point,
               'chemo_probabilities': chemo_probabilities,
               'radio_probabilities': radio_probabilities,
               'sequence_lengths': sequence_lengths,
               'death_flags': death_flags,
               'recovery_flags': recovery_flags,
               'patient_types': patient_types
               }

    return outputs

def simulate_intervention_assigned(simulation_params, num_time_steps, t_intervention, assigned_actions=None):
    """
    Core routine to generate simulation paths

    :param simulation_params:
    :param num_time_steps:
    :param assigned_actions:
    :return:
    """

    total_num_radio_treatments = 1
    total_num_chemo_treatments = 1

    radio_amt = np.array([2.0 for i in range(total_num_radio_treatments)])  # Gy
    radio_days = np.array([i + 1 for i in range(total_num_radio_treatments)])
    chemo_amt = [5.0 for i in range(total_num_chemo_treatments)]
    chemo_days = [(i + 1) * 7 for i in range(total_num_chemo_treatments)]

    # sort this
    chemo_idx = np.argsort(chemo_days)
    chemo_amt = np.array(chemo_amt)[chemo_idx]
    chemo_days = np.array(chemo_days)[chemo_idx]

    drug_half_life = 1  # one day half life for drugs

    # Unpack simulation parameters
    initial_stages = simulation_params['initial_stages']
    initial_volumes = simulation_params['initial_volumes']
    alphas = simulation_params['alpha']
    rhos = simulation_params['rho']
    betas = simulation_params['beta']
    beta_cs = simulation_params['beta_c']
    Ks = simulation_params['K']
    patient_types = simulation_params['patient_types']
    window_size = simulation_params['window_size']  # controls the lookback of the treatment assignment policy

    # Coefficients for treatment assignment probabilities
    chemo_sigmoid_intercepts = simulation_params['chemo_sigmoid_intercepts']
    radio_sigmoid_intercepts = simulation_params['radio_sigmoid_intercepts']
    chemo_sigmoid_betas = simulation_params['chemo_sigmoid_betas']
    radio_sigmoid_betas = simulation_params['radio_sigmoid_betas']

    num_patients = initial_stages.shape[0]

    # Commence Simulation
    cancer_volume = np.zeros((num_patients, num_time_steps))
    chemo_dosage = np.zeros((num_patients, num_time_steps))
    radio_dosage = np.zeros((num_patients, num_time_steps))
    chemo_application_point = np.zeros((num_patients, num_time_steps))
    radio_application_point = np.zeros((num_patients, num_time_steps))
    sequence_lengths = np.zeros(num_patients)
    death_flags = np.zeros((num_patients, num_time_steps))
    recovery_flags = np.zeros((num_patients, num_time_steps))
    chemo_probabilities = np.zeros((num_patients, num_time_steps))
    radio_probabilities = np.zeros((num_patients, num_time_steps))

    # Counterfactuals
    cancer_volume_counterfactual = np.zeros((num_patients, num_time_steps))
    chemo_dosage_counterfactual = np.zeros((num_patients, num_time_steps))
    radio_dosage_counterfactual = np.zeros((num_patients, num_time_steps))
    chemo_application_point_counterfactual = np.zeros((num_patients, num_time_steps))
    radio_application_point_counterfactual = np.zeros((num_patients, num_time_steps))

    noise_terms = 0.01 * np.random.randn(num_patients,
                                         num_time_steps)  # 5% cell variability
    recovery_rvs = np.random.rand(num_patients, num_time_steps)

    chemo_application_rvs = np.random.rand(num_patients, num_time_steps)
    radio_application_rvs = np.random.rand(num_patients, num_time_steps)

    # Run actual simulation
    for i in range(num_patients):

        logging.info("Simulating patient {} of {}".format(i + 1, num_patients))
        noise = noise_terms[i]

        # initial values
        cancer_volume[i, 0] = initial_volumes[i]
        alpha = alphas[i]
        beta = betas[i]
        beta_c = beta_cs[i]
        rho = rhos[i]
        K = Ks[i]


        # Setup cell volume
        b_death = False
        b_recover = False
        for t in range(1, num_time_steps):

            cancer_volume[i, t] = abs(cancer_volume[i, t - 1] * (1 + \
                                  + rho * np.log(K / cancer_volume[i, t - 1]) \
                                  - beta_c * chemo_dosage[i, t - 1] \
                                  - (alpha * radio_dosage[i, t - 1] + beta * radio_dosage[i, t - 1] ** 2) \
                                  + noise[t]))  # add noise to fit residuals
            
            if t >= t_intervention:
                if t == t_intervention:# Initialize whole row from cancer_volume. After t, we update using counterfactuals only.
                    cancer_volume_counterfactual[i, :] = cancer_volume[i, :]

                cancer_volume_counterfactual[i, t] = abs(cancer_volume_counterfactual[i, t - 1] * (1 + \
                                  + rho * np.log(K / cancer_volume_counterfactual[i, t - 1]) \
                                  - beta_c * chemo_dosage_counterfactual[i, t - 1] \
                                  - (alpha * radio_dosage_counterfactual[i, t - 1] + beta * radio_dosage_counterfactual[i, t - 1] ** 2) \
                                  + noise[t]))

            current_chemo_dose = 0.0
            previous_chemo_dose = 0.0 if t == 0 else chemo_dosage[i, t-1]

            # Action probabilities + death or recovery simulations
            cancer_volume_used = cancer_volume[i, max(t - window_size, 0):t]
            cancer_diameter_used = np.array([sim.calc_diameter(vol) for vol in cancer_volume_used]).mean()  # mean diameter over 15 days
            cancer_metric_used = cancer_diameter_used

            radio_prob = (1.0 / (1.0 + np.exp(-radio_sigmoid_betas[i]
                                            *(cancer_metric_used - radio_sigmoid_intercepts[i]))))
            chemo_prob = (1.0 / (1.0 + np.exp(- chemo_sigmoid_betas[i] *
                                                (cancer_metric_used - chemo_sigmoid_intercepts[i]))))

            chemo_probabilities[i, t] = chemo_prob
            radio_probabilities[i, t] = radio_prob

            # Action application
            if radio_application_rvs[i, t] < radio_prob :

                    radio_application_point[i, t] = 1
                    radio_dosage[i, t] = radio_amt[0]

            if chemo_application_rvs[i, t] < chemo_prob:

                # Apply chemo treatment
                chemo_application_point[i, t] = 1
                current_chemo_dose = 0#chemo_amt[0]

            # Update chemo dosage
            chemo_dosage[i, t] = previous_chemo_dose * np.exp(-np.log(2) / drug_half_life) + current_chemo_dose

                        # Update all counterfactuals
            if t >= t_intervention:
                if t == t_intervention:
                    previous_chemo_dose_counterfactual = previous_chemo_dose
                else:
                    previous_chemo_dose_counterfactual = chemo_dosage_counterfactual[i, t-1]
                
                chemo_application_point_counterfactual[i, t] = assigned_actions[i, t, 0]
                radio_application_point_counterfactual[i, t] = assigned_actions[i, t, 1]
                current_chemo_dose = 0#chemo_amt[0] if chemo_application_point_counterfactual[i, t] == 1 else 0
                radio_dosage_counterfactual[i, t] = radio_amt[0] if radio_application_point_counterfactual[i, t] == 1 else 0
                chemo_dosage_counterfactual[i, t] = previous_chemo_dose_counterfactual * np.exp(-np.log(2) / drug_half_life) + current_chemo_dose
            else:
                chemo_application_point_counterfactual[i, t] = chemo_application_point[i, t]
                radio_application_point_counterfactual[i, t] = radio_application_point[i, t]
                chemo_dosage_counterfactual[i, t] = chemo_dosage[i, t]
                radio_dosage_counterfactual[i, t] = radio_dosage[i, t]


        # Package outputs
        sequence_lengths[i] = int(t+1)
        death_flags[i, t] = 1 if b_death else 0
        recovery_flags[i, t] = 1 if b_recover else 0

    outputs = {'cancer_volume': cancer_volume,
               'cancer_volume_counterfactual': cancer_volume_counterfactual,
               'chemo_dosage_counterfactual': chemo_dosage_counterfactual,
               'radio_dosage_counterfactual': radio_dosage_counterfactual,
                'chemo_application_counterfactual': chemo_application_point_counterfactual,
                'radio_application_counterfactual': radio_application_point_counterfactual,
               'chemo_dosage': chemo_dosage,
               'radio_dosage': radio_dosage,
               'chemo_application': chemo_application_point,
               'radio_application': radio_application_point,
               'chemo_probabilities': chemo_probabilities,
               'radio_probabilities': radio_probabilities,
               'sequence_lengths': sequence_lengths,
               'death_flags': death_flags,
               'recovery_flags': recovery_flags,
               'patient_types': patient_types
               }

    return outputs

import numpy as np

# Initialize the array
assigned_actions = np.zeros((10000, 120, 2))

# Populate the array with random values
assigned_actions[:, :, 0] = np.random.choice([0, 1], size=(10000, 120), p=[0.99, 0.01])
assigned_actions[:, :, 1] = np.random.choice([0, 1], size=(10000, 120), p=[0.80, 0.20])

# Get the shape of the array
rows, cols, _ = assigned_actions.shape

# Loop to zero-out columns as specified
j = 0
for i in range(0, rows, 2000):
    end_row = i + 2000
    j += 1
    assigned_actions[i:end_row, j:, :] = 0

# Verify the changes
print(assigned_actions)

tumor_data = _generate_intervention_data(True, assigned_actions=assigned_actions)

# Checking that data looks right.

# Check that chemo_application_counterfactual agrees with chemo_application for the first 60 time points for all patients:
print(np.all(tumor_data['training_data']['chemo_application_counterfactual'][0:2000, 60:72].sum()==0))
print(np.all(tumor_data['training_data']['chemo_application_counterfactual'][2000:4000, 60:72].sum()==0))

tumor_data['training_data']['radio_application_counterfactual'][2000:4000, 2:20].mean()

# Plot the average cancer_volume and counterfactual_cancer_volume for the the time period, 1:120
import matplotlib.pyplot as plt

avg_cancer_volume = tumor_data['training_data']['cancer_volume'].mean(axis=0)
avg_cancer_volume_counterfactual = tumor_data['training_data']['cancer_volume_counterfactual'].mean(axis=0)
import matplotlib.pyplot as plt
plt.figure(figsize=(8, 8))
plt.scatter(range(120), avg_cancer_volume, label='cancer_volume')
plt.scatter(range(120), avg_cancer_volume_counterfactual, label='cancer_volume_counterfactual')
plt.show()

tumor_data['training_data']['radio_probabilities']