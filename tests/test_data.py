import numpy as np

def test_update_moving_stats():
    from src.utils.data import update_moving_stats
    batch_size = 100

    x_old = np.random.normal(size=(batch_size,))
    old_mean = np.mean(x_old, axis=0)
    old_mean_square = np.mean(x_old ** 2, axis=0)
    old_variance = x_old.var(0)

    x_new = np.random.normal(size=(batch_size,))

    x = np.hstack([x_old, x_new])
    mean = np.mean(x, axis=0)
    mean_square = np.mean(x ** 2, axis=0)
    variance = x.var(0)
    
    new_mean, new_mean_square, new_variance = update_moving_stats(
        x_new, old_mean, old_mean_square, old_variance, len(x_old), momentum=0
    )
    
    assert np.isclose(new_mean, mean, atol=1e-5)
    assert np.isclose(new_mean_square, mean_square, atol=1e-5)
    assert np.isclose(new_variance, variance, atol=1e-5)
    
if __name__ == "__main__":
    np.random.seed(0)

    test_update_moving_stats()
    print("test_update_moving_stats passed")