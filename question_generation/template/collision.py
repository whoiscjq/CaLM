from ananke.graphs import DAG
from ananke.models import binary_nested
import pandas as pd

def collision(graph, story, filter_invalid=False, types=["judge"]):
    results = []
    for t in types:
        assert t in ['judge', 'general', 'compute'], 'Not supported type: {}'.format(t)
        result = globals()['collision_' + t](
            graph=graph,
            story=story,
            filter_invalid=filter_invalid
        )
        for r in result:
            r["type"] = t
        results += result
    return results

def collision_judge(graph, story, filter_invalid=False):
    results = []
    N = len(graph.nodes)
    all_cases = story["all_cases"]

    # Does A directly affect B?
    for i in range(N):
        for j in range(N):
            if i == j: continue
            X = graph.nodes[i]
            Y = graph.nodes[j]
            Xn = story["semantic"][X]
            Yn = story["semantic"][Y]
            if not Xn or not Yn: continue
            ans = 'yes' if X in graph.graph[Y]["parent"] else 'no'
            results.append({
                "question": "Does {} directly affect {}?".format(Xn, Yn),
                "answer": ans
            })

    # Is A solely determined by B?
    for i in range(N):
        for j in range(N):
            if i == j: continue
            X = graph.nodes[i]
            Y = graph.nodes[j]
            Xn = story["semantic"][X]
            Yn = story["semantic"][Y]
            if not Xn or not Yn: continue
            if Y not in graph.graph[X]["parent"]:
                ans = 'invalid'
            elif len(graph.graph[X]["parent"]) > 1:
                ans = 'no'
            else:
                ans = 'yes'
            if ans == 'invalid' and filter_invalid: continue
            results.append({
                "question": "Is {} solely determined by {}?".format(Xn, Yn),
                "answer": ans
            })

    # Is A a cause of B?
    for i in range(N):
        for j in range(N):
            if i == j: continue
            X = graph.nodes[i]
            Y = graph.nodes[j]
            Xn = story["semantic"][X]
            Yn = story["semantic"][Y]
            if not Xn or not Yn: continue
            ans = 'yes' if graph.isCause(X, Y) else 'no'
            results.append({
                "question": "Is {} a cuase of {}?".format(Xn, Yn),
                "answer": ans
            })

    # Is A an effect of B?
    for i in range(N):
        for j in range(N):
            if i == j: continue
            X = graph.nodes[i]
            Y = graph.nodes[j]
            Xn = story["semantic"][X]
            Yn = story["semantic"][Y]
            if not Xn or not Yn: continue
            ans = 'yes' if graph.isCause(Y, X) else 'no'
            results.append({
                "question": "Is {} an effect of {}?".format(Xn, Yn),
                "answer": ans
            })

    #  if {A=a} and {B=b}, can we infer that {C=c} ? 
    if all_cases:
        for i in range(N-1):
            for j in range(i+1, N):
                for k in range(N):
                    if i == k or j == k:
                        continue
                    for P in [0, 1]:
                        for Q in [0, 1]:
                            for R in [0, 1]:
                                X = graph.nodes[i]
                                Y = graph.nodes[j]
                                Z = graph.nodes[k]
                                Xs = story["semantic"][X + str(P)]
                                Ys = story["semantic"][Y + str(Q)]
                                Zs = story["semantic"][Z + str(R)]
                                if not Xs or not Ys or not Zs: continue

                                ans = 'invalid'
                                for case in all_cases:
                                    if case[X] == P and case[Y] == Q:
                                        if case[Z] != R:
                                            ans = 'no'
                                        elif ans == 'invalid':
                                            ans = 'yes'
                                if ans == 'invalid' and filter_invalid: continue
                                results.append({
                                    "question": "If {} and {}, can we infer that {}?".format(Xs, Ys, Zs),
                                    "answer": ans
                                })

    #  if {A=a}, can we infer that {B=b} ?
    if all_cases:
        for i in range(N):
            for j in range(N):
                if i == j: continue
                for P in [0, 1]:
                    for Q in [0, 1]:
                        X = graph.nodes[i]
                        Y = graph.nodes[j]
                        Xs = story["semantic"][X + str(P)]
                        Ys = story["semantic"][Y + str(Q)]
                        if not Xs or not Ys: continue

                        ans = 'yes'
                        for case in all_cases:
                            if case[X] == P and case[Y] != Q:
                                ans = 'no'
                                break
                        
                        results.append({
                            "question": "If {}, can we infer that {}?".format(Xs, Ys),
                            "answer": ans
                        })

    #  Is {A=a} sufficient to cause {B=b} ?
    if all_cases:
        for i in range(N):
            for j in range(N):
                if i == j: continue
                for P in [0, 1]:
                    for Q in [0, 1]:
                        X = graph.nodes[i]
                        Y = graph.nodes[j]
                        Xn = story["semantic"][X + str(P) + '_noun']
                        Yn = story["semantic"][Y + str(Q) + '_noun']
                        if not Xn or not Yn: continue
                        if not graph.isCause(X, Y):  # X should be Y's cause
                            ans = 'invalid'
                        else:
                            ans = 'yes'
                            for case in all_cases:
                                if case[X] == P and case[Y] != Q:
                                    ans = 'no'
                                    break
                        if ans == 'invalid' and filter_invalid: continue
                        results.append({
                            "question": "Is {} sufficient to cause {}?".format(Xn, Yn),
                            "answer": ans
                        })

    #  Is {A=a} necessary to cause {B=b} ?
    if all_cases:
        for i in range(N):
            for j in range(N):
                if i == j: continue
                for P in [0, 1]:
                    for Q in [0, 1]:
                        X = graph.nodes[i]
                        Y = graph.nodes[j]
                        Xn = story["semantic"][X + str(P) + '_noun']
                        Yn = story["semantic"][Y + str(Q) + '_noun']
                        if not Xn or not Yn: continue
                        if not graph.isCause(X, Y):  # X should be Y's cause
                            ans = 'invalid'
                        else:
                            ans = 'yes'
                            for case in all_cases:
                                if case[Y] == Q and case[X] != P:
                                    ans = 'no'
                                    break
                        if ans == 'invalid' and filter_invalid: continue
                        results.append({
                            "question": "Is {} necessary to cause {}?".format(Xn, Yn),
                            "answer": ans
                        })

    # Does {A=a} always result in {B=b} ?
    if all_cases:
        for i in range(N):
            for j in range(N):
                if i == j: continue
                for P in [0, 1]:
                    for Q in [0, 1]:
                        X = graph.nodes[i]
                        Y = graph.nodes[j]
                        Xn = story["semantic"][X + str(P) + '_noun']
                        Yn = story["semantic"][Y + str(Q) + '_noun']
                        if not Xn or not Yn: continue
                        if not graph.isCause(X, Y):
                            ans = 'no'
                        else:
                            ans = 'yes'
                            for case in all_cases:
                                if case[X] == P and case[Y] != Q:
                                    ans = 'no'
                                    break
                        results.append({
                            "question": "Does {} always result in {}?".format(Xn, Yn),
                            "answer": ans
                        })

    # Does {A=a} directly cause {B=b}?
    if all_cases:
        for i in range(N):
            for j in range(N):
                if i == j: continue
                for P in [0, 1]:
                    for Q in [0, 1]:
                        X = graph.nodes[i]
                        Y = graph.nodes[j]
                        Xn = story["semantic"][X + str(P) + '_noun']
                        Yn = story["semantic"][Y + str(Q) + '_noun']
                        if not Xn or not Yn: continue
                        if X not in graph.graph[Y]["parent"]:
                            ans = 'no'
                        else:
                            ans = 'yes'
                            for case in all_cases:
                                if case[X] == P and case[Y] != Q:
                                    ans = 'no'
                                    break
                        results.append({
                            "question": "Does {} directly cause {}?".format(Xn, Yn),
                            "answer": ans
                        })

    # (counterfactual) if {A=a}, would still {B=b}?
    if all_cases:
        for X in graph.nodes:
            for Y in graph.nodes:
                if not graph.isCause(X, Y): continue
                for P in [0, 1]:
                    for Q in [0, 1]:
                        flag = False
                        for case in all_cases:
                            if case[X] == 1 - P and case[Y] == Q:
                                flag = True
                                break

                        if not flag: continue
                        Xn = story["semantic"][X + str(P) + '_past']
                        Yn = story["semantic"][Y + str(Q) + '_present']
                        if not Xn or not Yn: continue
                        right, wrong = False, False
                        for case in all_cases:
                            if case[X] == P:
                                if case[Y] == Q:
                                    right = True
                                else:
                                    wrong = True
                        if right and not wrong:
                            ans = 'yes'
                        elif not right and wrong:
                            ans = 'no'
                        elif right and wrong:
                            ans = 'not sure'
                        else:
                            ans = 'invalid'
                        if ans == 'invalid' and filter_invalid: continue
                        results.append({
                            "question": "If {}, would {}?".format(Xn, Yn),
                            "answer": ans
                        })

    return results


def collision_general(graph, story, filter_invalid=False):
    results = []
    N = len(graph.nodes)
    all_cases = story["all_cases"]

    # What directly affects A?
    for node in graph.nodes:
        X = story["semantic"][node]
        if not X: continue
        L = len(graph.graph[node]["parent"])
        if L == 1:
            p = graph.graph[node]["parent"][0]
            s = "Unobserved effect" if not story["semantic"][p] else story["semantic"][p]
            ans = "Based on the causal graph, {} directly affects {}.".format(s, X)
        elif L > 1:
            p = graph.graph[node]["parent"][0]
            s = "Unobserved effect" if not story["semantic"][p] else story["semantic"][p]
            for j in range(1, L):
                p = graph.graph[node]["parent"][j]
                Y = "Unobserved effect" if not story["semantic"][p] else story["semantic"][p]
                if j < L - 1:
                    s += ", {}".format(Y)
                else:
                    s += " and {}".format(Y)
            ans = "Based on the causal graph, {} directly affect {}.".format(s, X)
        else:
            ans = "Based on the causal graph, there's no direct cause of {}.".format(X)
        results.append({
            "question": "What factors directly affects {}?".format(X),
            "answer": ans
        })

    return results


def collision_compute(graph, story, filter_invalid=False):
    results = []
    if 'data' not in story:
        return results
    
    # prepare pd.DataFrame
    data = pd.DataFrame(story['data'])
    # print(data)
    
    # prepare DAG
    edges = [(e.split('->')[0], e.split('->')[1]) for e in story['edge']]
    g = DAG(vertices=story['node'], di_edges=edges)
    # dot = g.draw()
    # dot.render('test', format='svg')

    # load data
    X = binary_nested.process_data(df=data, count_variable='count')
    bnm = binary_nested.BinaryNestedModel(g)
    bnm = bnm.fit(X=X)
    # for key in bnm.fitted_params:
    #     print(type(key), type(key[0]), type(key[1]), type(key[2]))
    #     print(key, bnm.fitted_params[key])

    # marginal probability
    for i in story['node']:
        for P in [0, 1]:
            p = bnm.estimate(treatment_dict={}, outcome_dict={i: P}, check_identified=True)
            results.append({
                "question": "What is the probability that {}?".format(story['semantic'][i + str(P)]),
                "answer": "{:.6f}".format(p),
                "reason": "P({}={:d})={:.6f}".format(i, P, p),
                "detail_type": "marginal probability"
            })

    # intervention probability P(Y=y|do(X=x))
    for i in story['node']:
        for P in [0, 1]:
            for j in story['node']:
                for Q in [0, 1]:
                    if i == j: continue
                    Xn = story["semantic"].get(i + str(P) + "_past", None)
                    Yn = story["semantic"].get(j + str(Q), None)
                    if not Xn or not Yn: continue
                    p1 = bnm.estimate(treatment_dict={i:P}, outcome_dict={j:Q}, check_identified=True)
                    p2 = bnm.estimate(treatment_dict={i:1-P}, outcome_dict={j:Q}, check_identified=True)
                    ans = 'yes' if p1 - p2 > 1e-10 else 'no'
                    rel = '>' if p1 - p2 > 1e-10 else '<='
                    results.append({
                        "question": "If {}, would it be more likely that {}".format(Xn, Yn),
                        "answer": ans,
                        "reason": "P({}={:d}|do({}={:d})) - P({}={:d}|do({}={:d})) = {:.6f} - {:.6f} = {:.6f} {} 0".format(
                            j, Q, i, P, j, Q, i, 1-P, p1, p2, p1-p2, rel
                        ),
                        "detail_type": "intervention probability"
                    })


    # joint probability
    for i in story['node']:
        for P in [0, 1]:
            for j in story['node']:
                for Q in [0, 1]:
                    if i == j: continue
                    p = bnm.estimate(treatment_dict={}, outcome_dict={i: P, j: Q}, check_identified=True)
                    results.append({
                        "question": "What is the probability that {} AND {}?".format(
                            story['semantic'][i + str(P)],
                            story['semantic'][j + str(Q)]
                        ),
                        "answer": "{:.6f}".format(p),
                        "reason": "P({}={:d},{}={:d})={:.6f}".format(i, P, j, Q, p),
                        "detail_type": "joint probability"
                    })
    
    # conditional probablity
    for i in story['node']:
        for P in [0, 1]:
            for j in story['node']:
                for Q in [0, 1]:
                    if i == j: continue
                    p1 = bnm.estimate(treatment_dict={}, outcome_dict={i: P}, check_identified=True)
                    p2 = bnm.estimate(treatment_dict={}, outcome_dict={i: P, j: Q}, check_identified=True)
                    # P(X|Y) = P(X,Y)/P(Y)
                    results.append({
                        "question": "What is the probability that {} conditional on that {}?".format(
                            story['semantic'][j + str(Q)],
                            story['semantic'][i + str(P)]
                        ),
                        "answer": "{:.6f}".format(p2/p1),
                        "reason": "P({}={:d}|{}={:d})=P({}={:d},{}={:d})/P({}={:d})={:.6f}/{:.6f}={:.6f}".format(
                            j, Q, i, P, j, Q, i, P, i, P, p2, p1, p2/p1
                        ),
                        "detail_type": "conditional probability"
                    })

    return results
    
