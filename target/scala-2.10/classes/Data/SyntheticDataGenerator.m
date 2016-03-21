clear;clc;
%% Data Specs
n =10000;       % Sample Size
d =500;         % Vocabulary Size
k =2;           % Hidden Dimension
alpha0 =0.001;   % How mixed the topics are
n_test = 10;
expectedlen =100;

alphaprime=abs(randn(1,k));
alpha=alpha0*alphaprime/sum(alphaprime);
Aprime=zeros(n,k);
Aprime_test=zeros(n_test,k);
for j=1:k
  Aprime(:,j)=randg(alpha(j),n,1);
  Aprime_test(:,j)=randg(alpha(j),n_test,1);
end
ListNoGood = sum(Aprime,2)==0;
Aprime(ListNoGood,:)=1;

ListNoGood = sum(Aprime_test,2)==0;
Aprime_test(ListNoGood,:)=1;

A=bsxfun(@rdivide,Aprime,sum(Aprime,2));
A_test=bsxfun(@rdivide,Aprime_test,sum(Aprime_test,2));

Bprime=abs(randn(k,d));
beta=bsxfun(@rdivide,Bprime,sum(Bprime,2));


len = 2+ poissrnd(expectedlen,1,n);
Counts=mnrnd(len',A*beta);

len_test = 2+ poissrnd(expectedlen,1,n_test);
Counts_test = mnrnd(len_test',A_test*beta);

fid = fopen('datasets/synthetic/alpha.txt','wt');
for index_col = 1: length(alpha)
    fprintf(fid,'%.4f\t',alpha(index_col));
end
fclose(fid);


fid = fopen('datasets/synthetic/beta.txt','wt');
for index_row = 1: size(beta,1)
    for index_col = 1: size(beta,2)
        fprintf(fid,'%.4f\t',beta(index_row,index_col));
    end
    fprintf(fid,'\n');
end
fclose(fid);


% docID wordID:counts
fid = fopen('datasets/synthetic/samples_train_libsvm.txt','wt');
for index_doc = 1: n
    fprintf(fid, '%d ', index_doc-1);
    currentCount = Counts(index_doc,:);
    for index_word = 1:d
        thisCount = currentCount(index_word);
        if (thisCount~=0)
            fprintf(fid, '%d:%d ',index_word,currentCount(index_word));
        end
    end
    fprintf(fid, '\n');
end
fclose(fid);

fid = fopen('datasets/synthetic/samples_test_libsvm.txt','wt');
for index_doc = 1: n_test
    fprintf(fid, '%d ', index_doc-1);
    currentCount = Counts_test(index_doc,:);
    for index_word = 1:d
        thisCount = currentCount(index_word);
        if (thisCount~=0)
            fprintf(fid, '%d:%d ',index_word,currentCount(index_word));
        end
    end
    fprintf(fid, '\n');
end
fclose(fid);


%% write synthetic data in the bag of words format
% fid = fopen('datasets/synthetic/samples_train_DOK.txt','wt');
% for index_doc = 1 : n
% 	currentCount = Counts(index_doc,:);
% 	for index_word = 1:d
% 		thisCount = currentCount(index_word);
% 		if (thisCount ~= 0)
% 			fprintf(fid, '%d %d %d\n', index_doc-1, index_word-1, thisCount);
% 		end
% 	end
% end
% fclose(fid);

% fid = fopen('datasets/synthetic/samples_test_DOK.txt','wt');
% for index_doc = 1 : n_test
% 	currentCount = Counts(index_doc,:);
% 	for index_word = 1:d
% 		thisCount = currentCount(index_word);
% 		if (thisCount ~= 0)
% 			fprintf(fid, '%d %d %d\n', index_doc-1, index_word-1, thisCount);
% 		end
% 	end
% end
% fclose(fid);


