"""Thread budgets apply only to the isolated tracker, including affinity and CPU env."""
import io
import os
import unittest
from unittest.mock import Mock, patch

from mosaichunk_h3 import cpu_threads
from mosaichunk_h3.memory_controls import MemoryOptions
from mosaichunk_h3.occlusion_tracking import OcclusionTracker


class TrackerThreadTests(unittest.TestCase):
    def test_auto_uses_available_logical_threads_and_explicit_limits_are_capped(self):
        with patch.object(cpu_threads,'available_cpu_threads',return_value=24):
            self.assertEqual(cpu_threads.resolve_cpu_threads(0),24)
            self.assertEqual(cpu_threads.resolve_cpu_threads(8),8)
            self.assertEqual(cpu_threads.resolve_cpu_threads(64),24)

    def test_rejects_invalid_budgets_before_starting_a_worker(self):
        for value in (-1,257,True,1.5,'8',None):
            with self.subTest(value=value),self.assertRaises(ValueError):
                MemoryOptions(cpu_threads=value).validate()

    def test_process_cpu_count_or_affinity_is_preferred_to_machine_cpu_count(self):
        with patch.object(os,'process_cpu_count',return_value=6,create=True):
            self.assertEqual(cpu_threads.available_cpu_threads(),6)
        with patch.object(os,'process_cpu_count',return_value=None,create=True), \
             patch.object(os,'sched_getaffinity',return_value={0,2,4,6},create=True):
            self.assertEqual(cpu_threads.available_cpu_threads(),4)

    def test_missing_cpu_count_falls_back_to_one(self):
        with patch.object(os,'process_cpu_count',return_value=None,create=True), \
             patch.object(os,'sched_getaffinity',side_effect=OSError,create=True), \
             patch.object(os,'name','posix'),patch.object(os,'cpu_count',return_value=None):
            self.assertEqual(cpu_threads.available_cpu_threads(),1)

    def test_only_child_environment_and_pytorch_budget_are_configured(self):
        import torch
        parent_threads = torch.get_num_threads()
        parent_env = {key:os.environ.get(key) for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','CUDA_VISIBLE_DEVICES')}
        process = Mock(stdin=io.StringIO(),stdout=io.StringIO(''),stderr=io.StringIO(''))
        process.poll.return_value=0
        with patch.object(cpu_threads,'available_cpu_threads',return_value=24), \
             patch('subprocess.Popen',return_value=process) as spawn, \
             patch.object(OcclusionTracker,'wait',return_value={'ready':True,'cpu_threads':8}):
            with OcclusionTracker('unused',cpu_threads=8) as tracker:
                directory=tracker.temp.name
                command=spawn.call_args.args[0]
                env=spawn.call_args.kwargs['env']
                self.assertEqual(command[-1],'8')
                self.assertEqual(env['OMP_NUM_THREADS'],'8')
                self.assertEqual(env['MKL_NUM_THREADS'],'8')
                self.assertEqual(env['CUDA_VISIBLE_DEVICES'],'')
        self.assertFalse(os.path.exists(directory))
        self.assertEqual(torch.get_num_threads(),parent_threads)
        self.assertEqual({key:os.environ.get(key) for key in parent_env},parent_env)

    def test_worker_budget_mismatch_cleans_startup(self):
        process=Mock(stdin=io.StringIO(),stdout=io.StringIO(''),stderr=io.StringIO(''))
        process.poll.return_value=0
        with patch.object(cpu_threads,'available_cpu_threads',return_value=24), \
             patch('subprocess.Popen',return_value=process), \
             patch.object(OcclusionTracker,'wait',return_value={'ready':True,'cpu_threads':2}):
            tracker=OcclusionTracker('unused',cpu_threads=8)
            with self.assertRaisesRegex(RuntimeError,'does not match'):
                tracker.__enter__()
            self.assertFalse(os.path.exists(tracker.temp.name))


if __name__=='__main__':unittest.main()
