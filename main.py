import tensorflow as tf
import train_step.network

from train_step.network import cnnlite


def main():
    # This code is test for HOW ITERATOR IN TENSORFLOW WORKS.
    ## Only one calculation is needed, do like below.
    print("DEFINE START")
    cnnL = cnnlite()
    print("DEFINE END")

    print("INPUT LAYER DEF")
    input_layer = tf.random_uniform([100, 32, 32, 3])
    print("INPUT LAYER DEF END")
    print("CALCULATE INPUT LAYER")
    small_D_R = cnnL.forward(input_layer)
    print(small_D_R)

    ## But almost all time we need to calculate many models.
    input_layer1 = tf.random_uniform([1000000,32,32,3]) # Assum there exists very big many images
    ## In this case, when using pytorch, use Dataloader for make batch
    print("CALCULATE BIG INPUT LAYER")
    result = cnnL.forward(input_layer1)
    print( result )

    sess = tf.Session()
    sess.run(small_D_R)

    iterator = input_layer1.make_initializable_iterator()
    next_element = iterator.get_next()

    print("END")


if __name__=='__main__':
    main()