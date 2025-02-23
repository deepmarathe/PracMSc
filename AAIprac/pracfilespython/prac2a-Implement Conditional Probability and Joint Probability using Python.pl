% Facts: Define symptoms
symptom(fever).
symptom(cough).
symptom(sore_throat).
symptom(body_aches).
symptom(runny_nose).
symptom(headache).
symptom(fatigue).

% Facts: Define possible illnesses
condition(cold).
condition(flu).
condition(strep_throat).

% Rules: Diagnosing based on the presence of symptoms
diagnose(cold) :-
    symptom(runny_nose), 
    symptom(cough), 
    symptom(sore_throat), 
    \+ symptom(fever). % Absence of fever

diagnose(flu) :-
    symptom(fever), 
    symptom(cough), 
    symptom(body_aches),
    symptom(headache), 
    symptom(fatigue).

diagnose(strep_throat) :-
    symptom(sore_throat), 
    symptom(fever), 
    \+ symptom(cough). % Absence of cough

% Alternative: Diagnosing based on rules covering all possible symptoms
diagnose(unknown) :-
    \+ diagnose(cold),
    \+ diagnose(flu),
    \+ diagnose(strep_throat).

% Queries: Example of how to diagnose
% ?- diagnose(Condition).
% Output: Condition = flu. (if the symptoms match the flu criteria)

% Assuming the patient has the following symptoms:
% symptom(fever).
% symptom(cough).
% symptom(body_aches).
% symptom(headache).
% symptom(fatigue).

% You can ask Prolog:
?- diagnose(Condition).
