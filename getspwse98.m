% getspwse98 gets the specificity, sp, and threshold, T, after setting a bar of 98% sensitivity for the machine learning model
function [sp,T] = getspwse98(Xalg,Yalg,T)
placese = find(Yalg >= .98);
if isempty(placese)
    placese = length(Yalg);
else
    placese = placese(1);
end
sp = 1-Xalg(placese);
T = T(placese);
end
