

public class Main { // 一个java中只能有一个类是public的。
    public static void main(String[] args) {
        System.out.println("hello world");
        B test = new B();
        test.doSomething();
    }
}
// 类变量是所有该类的实例化对象所共有的资源，其中一个对象将它值改变，其他对象得到的就是改变后的结果；
// 而实例变量则属对象私有，某一个对象将其所包含的实例变量的值改变，不影响其他对象中实例变量的值；
abstract class A {
    public static int x=10; // 类变量
    public int y =10; // 实例变量
    public A(){
        System.out.println("A is here");}
    abstract void doSomething();
}


class B extends A {
    public B(){
        super();
        System.out.println("This is B");
    }
    // 下方3个方法刚好对应调用变量的方式。this是当前对象；myparent是新建对象引用；
    // A是类，但注意了only call类变量(statci)，不能call实例变量。但这种方法只适用于基础数据，否则容易memory leak.
    @Override
    public void doSomething(){
        B myparent = new B();
        System.out.println("X=" + myparent.x); //10 // this./A.都是10
        this.x=28;
        System.out.println("X=" + this.x); //28 // myparent./A.都是28
        A.x = 32;
        System.out.println("X=" + A.x); // 32 // myparent/this都是32
    }
}
// static全局变量只初使化一次，防止在其他文件单元中被引用;
// static局部变量只被初始化一次，下一次依据上一次结果值；