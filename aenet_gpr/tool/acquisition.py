import numpy as np
import random


def acquisition(train_images, candidates, mode='min_energy', objective='min'):
    """
    Acquisition function class.
    This function is in charge of ordering a list of Atoms candidates to
    be evaluated in order to minimize the number of function queries to get
    to the converged solution.

    Parameters
    ----------
    train_images: list
        List of Atoms containing the previously evaluated structures.
    candidates: list
        Unordered list of Atoms with the candidates to be evaluated.
    mode: string
        Name of the acquisition function in charge of ordering the candidates.
        Modes available are:
            - 'energy': Sort candidates by predicted energy.
            - 'uncertainty': Sort candidates by predicted uncertainty.
            - 'random': Randomly shuffles the candidates list.
            - 'ucb': Sort candidates by predicted energy plus uncertainty.
            - 'lcb': Sort candidates by predicted energy minus uncertainty.
    objective: string
         Objective of the acquisition function. Options:
            - 'min': If our goal is to minimize the target function.
            - 'max': If our goal is to maximize the target function.


    Returns
    -------
    Sorted candidates: list
        Ordered list of Atoms with the candidates to be evaluated.
    """

    x = []
    y = []
    pred_x = []
    pred_y = []
    pred_fmax = []
    pred_func = []

    # Gather all required information to decide the order of the candidates.
    for i in candidates:
        pred_x.append(i.get_positions().reshape(-1))
        pred_y.append(i.get_potential_energy())
        if mode == 'uncertainty' or mode == 'ucb' or mode == 'lcb':
            # pred_unc.append(i.calc.results['unc_energy'])
            f_unc = i.calc.results['unc_forces']
            pred_func.append(np.linalg.norm(f_unc, axis=1).max())

        if mode == 'fmax':
            force = i.get_forces()
            pred_fmax.append(np.linalg.norm(force, axis=1).max())

    for i in train_images:
        x.append(i.get_positions().reshape(-1))
        y.append(i.get_potential_energy())

    implemented_acq = ['energy', 'fmax', 'uncertainty', 'ucb', 'lcb', 'random']

    if mode not in implemented_acq:
        msg = 'The selected acquisition function is not implemented. ' \
              'Implemented are: ' + str(implemented_acq)
        raise NotImplementedError(msg)

    if mode == 'energy':
        if objective == 'min':
            score_index = np.argsort(pred_y)
        if objective == 'max':
            score_index = list(reversed(np.argsort(pred_y)))

    if mode == 'uncertainty':
        if objective == 'min':
            score_index = np.argsort(pred_func)
        if objective == 'max':
            score_index = list(reversed(np.argsort(pred_func)))

    if mode == 'fmax':
        if objective == 'min':
            score_index = np.argsort(pred_fmax)
        if objective == 'max':
            score_index = list(reversed(np.argsort(pred_fmax)))

    if mode == 'ucb':
        e_plus_u = np.array(pred_y) + np.array(pred_func)
        if objective == 'min':
            score_index = np.argsort(e_plus_u)
        if objective == 'max':
            score_index = list(reversed(np.argsort(e_plus_u)))

    if mode == 'lcb':
        e_minus_u = np.array(pred_y) - np.array(pred_func)
        if objective == 'min':
            score_index = np.argsort(e_minus_u)
        if objective == 'max':
            score_index = list(reversed(np.argsort(e_minus_u)))

    if mode == 'random':
        ordered_index = list(range(len(candidates)))
        score_index = random.sample(ordered_index, len(ordered_index))

    # Order candidates (from best to worst candidates):
    ordered_images = []
    for i in score_index:
        ordered_images.append(candidates[i])

    return ordered_images
