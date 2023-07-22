using namespace std;
#include <vector>
#include <iostream>
class Solution{
public:
    int main(){
        vector<int> vect = { 10, 20, 30 };
        int k = 2;
        distributeCookies(vect, k);
        return 0;
    }
    int distributeCookies(vector<int> & cookies, int k){
        int n = cookies.size(); //This just represents the number of bags of cookkies we need to distribute.
        vector<vector<int> > dp = vector<vector<int> > (k+1, vector<int>(1ll << n, INT_MAX)); //What does this do?
        for(int i = 0 ; i < dp.size(); i ++){
            for(int j = 0; j < dp[0].size(); j++){
                cout << dp[i][j];
            }
        }
        //Yeah, I wasn't able to solve this oops mbmbmbmb.
        return 0;
    }
};
