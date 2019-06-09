#include <iostream>
using namespace std;

int main() {
int a, b, c, n;
cin>>n;
a=0;
b=1;
for(int i=0; i<n-2; i++)
{
    c=a+b;
    cout<<c<<endl;
    a=b;
    b=c;
}

return 0;
}

