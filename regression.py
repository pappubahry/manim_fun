import manim as mn
import numpy as np
import warnings

warnings.simplefilter("error")

# Caching causes problems:
mn.config.disable_caching = True

class LinearRegressionResiduals(mn.Scene):
    def construct(self):
        pca_color = mn.rgb_to_color([ 27, 158, 119])
        ols_color = mn.rgb_to_color([117, 112, 179])
        pca_text_color = mn.rgb_to_color([ 50, 208, 184])
        ols_text_color = mn.rgb_to_color([182, 180, 230])

        x = np.array([10.3, 9.6, 10.4, 10.1, 9.4, 9.8, 10.1, 9.5, 10.0, 9.7, 9.3, 10.2, 9.8, 9.8, 9.2])
        y = np.array([ 4.9, 5.8,  5.2, 5.7,  6.0, 3.5,  6.2, 6.3,  4.8, 5.5, 4.9,  4.3, 4.5, 6.0, 3.4])
        
        x_mean = np.mean(x)
        y_mean = np.mean(y)
        x_min = np.min(x)
        x_max = np.max(x)
        y_min = np.min(y)
        y_max = np.max(y)

        x_range = x_max - x_min
        y_range = y_max - y_min

        # We want x_range and y_range to be the same,
        # so that the residuals are visually drawn
        # perpendicular to the PCA regression line.
        delta_range = x_range - y_range
        if delta_range > 0:
            y_max += 0.5*delta_range
            y_min -= 0.5*delta_range
        else:
            x_max -= 0.5*delta_range
            x_min += 0.5*delta_range

        buffer = 0.1*x_range
        
        x_min -= buffer
        x_max += buffer
        y_min -= buffer
        y_max += buffer

        # Compute the angles of the regression lines:
        cov = np.cov(x, y)
        target_slope_ols = cov[0, 1] / np.var(x)
        target_theta_ols = np.arctan(target_slope_ols)

        evals, evecs = np.linalg.eig(cov)
        evec = evecs[:, np.argmax(evals)]
        target_theta_pca = np.arctan2(evec[1], evec[0])
        if target_theta_pca < 0:
            target_theta_pca += mn.TAU / 2

        axes = mn.Axes(
            x_range = [x_min, x_max, 1],
            y_range = [y_min, y_max, 1],
            x_length = 7,
            y_length = 7,
            tips = False,
            axis_config = {"include_numbers": True}
        ).move_to([-3, 0., 0.])
        self.add(axes)
        
        time_tracker = mn.ValueTracker(0)
        self.theta = 0.

        def update_slope_pca(mob):
            # Idea is smootherstep decaying of the amplitude of a sine wave,
            # for pretty animation purposes
            t = time_tracker.get_value()
            st = mn.smootherstep(t)
            self.theta = target_theta_pca * (1 - (1 - st) * np.cos(t * mn.TAU))

        controller = mn.VectorizedPoint().set_opacity(0)
        controller.add_updater(update_slope_pca)

        points = []
        for x_val, y_val in zip(x, y):
            dot = mn.Dot(axes.coords_to_point(x_val, y_val), color=mn.YELLOW)
            points.append((x_val, y_val))
            self.add(dot)
        self.wait(1.)

        # Initial line parameters
        line_start = axes.coords_to_point(x_min, y_mean)
        line_end = axes.coords_to_point(x_max, y_mean)
        regression_line = mn.Line(line_start, line_end, color=pca_color, stroke_width=6)

        def update_regression_line(mob):
            m = np.tan(self.theta)
            b = y_mean - m * x_mean
            new_start = axes.coords_to_point(x_min, m*x_min + b)
            new_end   = axes.coords_to_point(x_max, m*x_max + b)
            mob.put_start_and_end_on(new_start, new_end)
        
        regression_line.add_updater(update_regression_line)
        self.play(mn.Create(regression_line), run_time = 1.)

        residual_lines = []

        # Store the end-points of each residual for use in the transition from
        # PCA to OLS:
        residual_ends = []
        
        def residual_color(d):
            # White-red colour scale
            d_color_scale_max = 2.5
            gb = max(0, int(255 * (1 - np.sqrt(d / d_color_scale_max))))
            return mn.rgb_to_color([255, gb, gb])

        for i, (x_val, y_val) in enumerate(points):
            start_point = axes.coords_to_point(x_val, y_val)
            end_point = axes.coords_to_point(x_val, y_mean)
            d = np.abs(y_val - y_mean)

            residual_line = mn.Line(start_point, end_point, color=residual_color(d), stroke_width=2)

            def update_residual_pca(mob, i_local = i, x_local = x_val, y_local = y_val):
                c = np.cos(self.theta)
                s = np.sin(self.theta)
                v = np.array([ c, s])
                u = np.array([-s, c])
                # Line is s*x - c*y = constant
                constant = s*x_mean - c*y_mean
                # Assume line isn't vertical, so we can divide by c
                p0 = np.array([0., -constant / c])
                p = np.array([x_local, y_local])
                t = np.dot(v, p - p0)
                s = np.dot(u, p - p0)
                end_pt = p0 + t*v
                new_start = axes.coords_to_point(x_local, y_local)
                new_end = axes.coords_to_point(*end_pt)

                mob.set_color(residual_color(np.abs(s)))
                mob.put_start_and_end_on(new_start, new_end)

                residual_ends[i_local] = end_pt
            
            residual_line.add_updater(update_residual_pca)
            residual_lines.append(residual_line)
            residual_ends.append(np.array([]))
        
        self.play(*[mn.Create(line) for line in residual_lines], run_time = 1.)
        
        self.wait(0.1)
        # Only add the controller now, otherwise the time tracker starts while the residual
        # lines are being drawn.
        self.add(controller)

        self.play(time_tracker.animate.set_value(1.), run_time=5, rate_func=mn.linear)

        # PCA animation finished; remove all updaters
        for line in residual_lines:
            for updater in line.get_updaters():
                line.remove_updater(updater)
        
        pca_label = mn.Tex(
            "Line parallel to first\\\\principal component",
            color = pca_text_color,
            font_size = 48,
            tex_environment = "flushleft")
        pca_label.move_to([3.555, 2., 0.]).to_edge(mn.RIGHT)
        self.play(mn.Write(pca_label))

        self.wait(1.)

        # Now transition the residual lines to vertical for the OLS

        angle = self.theta
        m = np.tan(angle)
        b = y_mean - m * x_mean

        for i, (x_val, y_val) in enumerate(points):
            x0_end, y0_end = residual_ends[i]
            y_reg = m*x_val + b

            def update_residual_transition(mob, i_loc = i, x_loc = x_val, y_loc = y_val, x0_end_loc = x0_end, y0_end_loc = y0_end, y_reg_loc = y_reg):
                t = time_tracker.get_value()
                st = mn.smoothstep(t)
                start = np.array([x_loc, y_loc])
                start_point = axes.coords_to_point(*start)

                end_x = (1 - st) * x0_end_loc + st * x_loc
                end_y = (1 - st) * y0_end_loc + st * y_reg_loc
                end = np.array([end_x, end_y])
                end_point = axes.coords_to_point(*end)

                mob.put_start_and_end_on(start_point, end_point)

                d = np.linalg.norm((end - start))
                mob.set_color(residual_color(d))
            
            residual_lines[i].add_updater(update_residual_transition)

        controller.remove_updater(update_slope_pca)
        time_tracker.set_value(0.)
        self.play(time_tracker.animate.set_value(1.), run_time=2., rate_func=mn.linear)

        # Transition to OLS finished; remove updaters
        for line in residual_lines:
            for updater in line.get_updaters():
                line.remove_updater(updater)

        
        # Now the OLS animation

        def update_slope_ols(mob):
            t = time_tracker.get_value()
            st = mn.smootherstep(t)
            self.theta = target_theta_pca + (target_theta_ols - target_theta_pca) * (1 - (1 - st) * np.cos(t * mn.TAU))

        for i, (x_val, y_val) in enumerate(points):
            def update_residual_ols(mob, x_local = x_val, y_local = y_val):
                m = np.tan(self.theta)
                b = y_mean - m * x_mean
                y_reg = m*x_local + b

                new_start = axes.coords_to_point(x_local, y_local)
                new_end = axes.coords_to_point(x_local, y_reg)

                mob.set_color(residual_color(np.abs(y_reg - y_local)))
                mob.put_start_and_end_on(new_start, new_end)
            
            residual_lines[i].add_updater(update_residual_ols)
        
        # Temporarily remove the updater for the regression line
        # while we copy it
        regression_line.remove_updater(update_regression_line)
        static_line = regression_line.copy()

        time_tracker.set_value(0.)

        regression_line.add_updater(update_regression_line)
        controller.add_updater(update_slope_ols)

        self.add(static_line)

        regression_line.set_color(ols_color)

        self.play(time_tracker.animate.set_value(1.), run_time=5, rate_func=mn.linear)

        ols_label = mn.Tex(
            "Ordinary least squares\\\\regression line",
            color = ols_text_color,
            font_size = 48,
            tex_environment = "flushleft")
        ols_label.move_to([3.555, 0., 0.]).to_edge(mn.RIGHT)
        self.play(mn.Write(ols_label))

        self.wait(0.5)

        self.play(*[mn.FadeOut(line) for line in residual_lines], run_time = 0.5)

        self.wait(2.)
