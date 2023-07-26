function [r,c,v] = GMOSSparseIDs(D,n,dTor)
Dn = D*n;

if n > 1
    % multiple shooting
    r = zeros(2*n*D^2+((dTor+2)*n+(dTor-1))*D+dTor,1); % row indices
    c = zeros(2*n*D^2+((dTor+2)*n+(dTor-1))*D+dTor,1); % column indices
    v = zeros(2*n*D^2+((dTor+2)*n+(dTor-1))*D+dTor,1); % values to store
else
    % single shooting
    r = zeros(D^2+(2*dTor+1)*D+dTor,1);
    c = zeros(D^2+(2*dTor+1)*D+dTor,1);
    v = zeros(D^2+(2*dTor+1)*D+dTor,1);
end
for kk = 1:n
    % Index
    idx = D*(kk-1)+1:D*kk;

    if n > 1
        % Continuity - multiple shooting
        r(D^2*(kk-1)+1:D^2*kk) = repmat(idx',D,1);
        c(D^2*(kk-1)+1:D^2*kk) = reshape(repmat(idx,D,1),D^2,1);
    end
    if kk == 1
        % Quasi-periodicity
        if n > 1
            % mutltiple shooting
            ad_id = (2*n-1)*D^2+(n-1)*D;
            r(ad_id+(1:D^2)) = repmat((1:D)',D,1);
            c(ad_id+(1:D^2)) = reshape(repmat(((n-1)*D+1):Dn,D,1),D^2,1);

            for jj = 1:dTor
                ad_id = 2*n*D^2+(n-1)*D + (jj-1)*D;
                r(ad_id+(1:D)) = (1:D)';
                c(ad_id+(1:D)) = (Dn+jj)*ones(D,1);
            end
        else
            % single shooting
            r(1:D^2) = repmat((1:D)',D,1);
            c(1:D^2) = reshape(repmat(1:D,D,1),D^2,1);

            for jj = 1:dTor
                ad_id = D^2 + (jj-1)*D;
                r(ad_id+(1:D)) = (1:D)';
                c(ad_id+(1:D)) = (D+jj)*ones(D,1);
            end
        end
    else
        % Continutiy - multiple shooting
        ad_id = n*D^2;
        r(ad_id+(D^2*(kk-2)+1:D^2*(kk-1))) = repmat(idx',D,1);
        c(ad_id+(D^2*(kk-2)+1:D^2*(kk-1))) = reshape(repmat(D*(kk-2)+1:D*(kk-1),D,1),D^2,1);

        ad_id = (2*n-1)*D^2;
        r(ad_id+(D*(kk-2)+1:D*(kk-1))) = idx';
        c(ad_id+(D*(kk-2)+1:D*(kk-1))) = (Dn+1)*ones(D,1);
    end
end

% Phase Constraints
if n > 1
    % multiple shooting
    ad_id = 2*n*D^2 + (n+dTor-1)*D - Dn;
else
    % single shooting
    ad_id = D^2 + (dTor-1)*D;
end
for jj = 1:dTor
    ad_id = ad_id + Dn;
    r(ad_id+(1:Dn)) = (Dn+jj)*ones(Dn,1);
    c(ad_id+(1:Dn)) = 1:Dn;
end

% Pseudo-arclength continuation
if n > 1
    % multiple shooting
    ad_id = 2*n*D^2 + ((dTor+1)*n+dTor-1)*D;
    r(ad_id+(1:Dn)) = (Dn+dTor+1)*ones(Dn,1);
    c(ad_id+(1:Dn)) = 1:Dn;

    ad_id = 2*n*D^2 + ((dTor+2)*n+dTor-1)*D;
    r(ad_id+(1:dTor)) = (Dn+dTor+1)*ones(dTor,1);
    c(ad_id+(1:dTor)) = Dn+(1:dTor);
else
    % single shooting
    ad_id = D^2+2*dTor*D;
    r(ad_id+(1:D)) = (D+dTor+1)*ones(D,1);
    c(ad_id+(1:D)) = 1:D;

    ad_id = D^2+(2*dTor+1)*D;
    r(ad_id+(1:dTor)) = (D+dTor+1)*ones(dTor,1);
    c(ad_id+(1:dTor)) = D+(1:dTor);
end
end