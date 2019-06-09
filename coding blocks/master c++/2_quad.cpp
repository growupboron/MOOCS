#include<iostream>
#include<math.h>
using namespace std;
int main() {
	float a, b, c, d, D, r1, r2;
	cin>>a;
	cin>>b;
	cin>>c;
	D=b*b-4*a*c;
	d=sqrt(D);
		if (D>0)
		{
			r1=((-1*b)+d)/(2*a);
			r2=((-1*b)-d)/(2*a);
			cout<<"Real and Distinct"<<endl<<r2<<" "<<r1;
		}
		else if (D=0)
		{
			r1=((-1*b)+d)/(2*a);
			//r2=(-b-D)/2a;
			cout<<"Real and Equal"<<endl<<r1<<" "<<r1;
		}
		else
		{
			cout<<"Imaginary";
		}

	return 0;
}
