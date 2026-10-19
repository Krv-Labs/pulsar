//! Point-level views of a Ball Mapper (`pulsar/analysis/hooks.py`).

use ndarray::Array2;
use pulsar_core::BallMapper;

/// For each of `n` points, the id of its first ball, or `-1` if no ball
/// covers it.
pub fn label_points(ball_mapper: &BallMapper, n: usize) -> Vec<i64> {
    let mut labels = vec![-1_i64; n];
    for (ball_id, members) in ball_mapper.nodes.iter().enumerate() {
        for &pt in members {
            if pt < n && labels[pt] == -1 {
                labels[pt] = ball_id as i64;
            }
        }
    }
    labels
}

/// Dense `n × n_balls` membership matrix: `m[i, b] == 1` when point `i`
/// belongs to ball `b`.
pub fn membership_matrix(ball_mapper: &BallMapper, n: usize) -> Array2<u8> {
    let mut m = Array2::<u8>::zeros((n, ball_mapper.nodes.len()));
    for (ball_id, members) in ball_mapper.nodes.iter().enumerate() {
        for &pt in members {
            if pt < n {
                m[[pt, ball_id]] = 1;
            }
        }
    }
    m
}

/// Points no ball covers.
pub fn unclustered_points(ball_mapper: &BallMapper, n: usize) -> Vec<usize> {
    let mut covered = vec![false; n];
    for members in &ball_mapper.nodes {
        for &pt in members {
            if pt < n {
                covered[pt] = true;
            }
        }
    }
    (0..n).filter(|&i| !covered[i]).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn first_ball_wins_and_uncovered_points_are_reported() {
        let bm = BallMapper {
            eps: 1.0,
            nodes: vec![vec![0, 1, 2], vec![2, 3]],
            edges: vec![(0, 1)],
        };
        assert_eq!(label_points(&bm, 5), vec![0, 0, 0, 1, -1]);
        let m = membership_matrix(&bm, 5);
        assert_eq!(m.dim(), (5, 2));
        assert_eq!(m.row(2).to_vec(), vec![1, 1]);
        assert_eq!(m.row(4).to_vec(), vec![0, 0]);
        assert_eq!(unclustered_points(&bm, 5), vec![4]);
    }
}
