#!/usr/bin/env python3
"""Modified realtime transcription that streams to web browser."""

import multiprocessing
import threading
import json
import asyncio
import websockets
import time

from absl import app
from absl import flags
import attr
from colorama import Fore
from colorama import Style
import audio_recorder
import tflite_model
import numpy as np

flags.DEFINE_string('model_path', 'onsets_frames_wavinput.tflite',
                    'File path of TFlite model.')
flags.DEFINE_string('mic', None, 'Optional: Input source microphone ID.')
flags.DEFINE_float('mic_amplify', 30.0, 'Multiply raw audio mic input')
flags.DEFINE_string(
    'wav_file', None,
    'If specified, will decode the first 10 seconds of this wav file.')
flags.DEFINE_integer(
    'sample_rate_hz', 16000,
    'Sample Rate. The model expects 16000. However, some microphones do not '
    'support sampling at this rate. In that case use --sample_rate_hz 48000 and'
    'the code will automatically downsample to 16000')
flags.DEFINE_integer('websocket_port', 8765, 'WebSocket server port')
flags.DEFINE_boolean('web_output', True, 'Enable web output via WebSocket')
FLAGS = flags.FLAGS

# Global WebSocket connections
websocket_clients = set()

class TfLiteWorker(multiprocessing.Process):
  """Process for executing TFLite inference."""

  def __init__(self, model_path, task_queue, result_queue):
    multiprocessing.Process.__init__(self)
    self._model_path = model_path
    self._task_queue = task_queue
    self._result_queue = result_queue
    self._model = None

  def setup(self):
    if self._model is not None:
      return

    self._model = tflite_model.Model(model_path=self._model_path)

  def run(self):
    self.setup()
    while True:
      task = self._task_queue.get()
      if task is None:
        self._task_queue.task_done()
        return
      task(self._model)
      self._task_queue.task_done()
      self._result_queue.put(task)


@attr.s
class AudioChunk(object):
  serial = attr.ib()
  samples = attr.ib(repr=lambda w: '{} {}'.format(w.shape, w.dtype))


class AudioQueue(object):
  """Audio queue."""

  def __init__(self, callback, audio_device_index, sample_rate_hz,
               model_sample_rate, frame_length, overlap):
    # Initialize recorder.
    downsample_factor = sample_rate_hz / model_sample_rate
    self._recorder = audio_recorder.AudioRecorder(
        sample_rate_hz,
        downsample_factor=downsample_factor,
        device_index=audio_device_index)

    self._frame_length = frame_length
    self._overlap = overlap

    self._audio_buffer = np.array([], dtype=np.int16).reshape(0, 1)
    self._chunk_counter = 0
    self._callback = callback

  def start(self):
    """Start processing the queue."""
    with self._recorder:
      timed_out = False
      while not timed_out:
        assert self._recorder.is_active
        new_audio = self._recorder.get_audio(self._frame_length -
                                             len(self._audio_buffer))
        audio_samples = np.concatenate(
            (self._audio_buffer, new_audio[0] * FLAGS.mic_amplify))

        # Extract overlapping
        first_unused_byte = 0
        for pos in range(0, audio_samples.shape[0] - self._frame_length,
                         self._frame_length - self._overlap):
          self._callback(
              AudioChunk(self._chunk_counter,
                         audio_samples[pos:pos + self._frame_length]))
          self._chunk_counter += 1
          first_unused_byte = pos + self._frame_length

        # Keep the remaining bytes for next time
        self._audio_buffer = audio_samples[first_unused_byte:]


class OnsetsTask(object):
  """Inference task."""

  def __init__(self, audio_chunk: AudioChunk):
    self.audio_chunk = audio_chunk
    self.result = None

  def __call__(self, model):
    samples = self.audio_chunk.samples[:, 0]
    self.result = model.infer(samples)
    self.timestep = model.get_timestep()


def note_number_to_name(note_number):
  """Convert note index to note name - matches terminal output mapping."""
  # This matches the notename_color function: A, A#, B, C, C#, D, D#, E, F, F#, G, G#
  note_names = ['A', 'A#', 'B', 'C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#']
  return note_names[note_number % 12]


def notename_color(n, space):
  """Get colored note name for console output."""
  if space:
    return [' ', '  ', ' ', ' ', '  ', ' ', '  ', ' ', ' ', '  ', ' ',
            '  '][n % 12]
  return [
      Fore.BLUE + 'A' + Style.RESET_ALL,
      Fore.LIGHTBLUE_EX + 'A#' + Style.RESET_ALL,
      Fore.GREEN + 'B' + Style.RESET_ALL,
      Fore.CYAN + 'C' + Style.RESET_ALL,
      Fore.LIGHTCYAN_EX + 'C#' + Style.RESET_ALL,
      Fore.RED + 'D' + Style.RESET_ALL,
      Fore.LIGHTRED_EX + 'D#' + Style.RESET_ALL,
      Fore.YELLOW + 'E' + Style.RESET_ALL,
      Fore.WHITE + 'F' + Style.RESET_ALL,
      Fore.LIGHTBLACK_EX + 'F#' + Style.RESET_ALL,
      Fore.MAGENTA + 'G' + Style.RESET_ALL,
      Fore.LIGHTMAGENTA_EX + 'G#' + Style.RESET_ALL,
  ][n % 12]


async def websocket_handler(websocket):
    """Handle WebSocket connections."""
    websocket_clients.add(websocket)
    print(f"WebSocket client connected. Total clients: {len(websocket_clients)}")
    try:
        await websocket.wait_closed()
    finally:
        websocket_clients.remove(websocket)
        print(f"WebSocket client disconnected. Total clients: {len(websocket_clients)}")


async def broadcast_notes(note_data):
    """Broadcast note data to all connected WebSocket clients."""
    if websocket_clients:
        message = json.dumps(note_data)
        disconnected = set()
        
        for client in websocket_clients:
            try:
                await client.send(message)
            except websockets.exceptions.ConnectionClosed:
                disconnected.add(client)
        
        # Remove disconnected clients
        websocket_clients.difference_update(disconnected)


def result_collector(result_queue, websocket_loop):
  """Collect and display results, also send to WebSocket."""
  print('Listening to results..')
  
  while True:
    result = result_queue.get()
    serial = result.audio_chunk.serial
    result_roll = result.result
    
    if serial > 0:
      result_roll = result_roll[4:]
    
    # Collect active notes for this frame
    active_notes = []
    console_output = []
    
    for notes in result_roll:
      frame_notes = []
      for i in range(6, len(notes) - 6):
        note = notes[i]
        is_frame = note[0] > 0.3  # Frame threshold
        is_onset = note[1] > 0.5  # Onset threshold
        velocity = note[3] if len(note) > 3 else 1.0
        
        if is_frame or is_onset:
          note_name = note_number_to_name(i)
          # Correct octave calculation for piano range (A0 = MIDI 21)
          midi_note = i + 21
          octave = (midi_note - 12) // 12  # Standard MIDI octave calculation
          
          frame_notes.append({
            'note': note_name,
            'octave': octave,
            'midi_note': midi_note,
            'velocity': float(velocity),
            'frame_prob': float(note[0]),
            'onset_prob': float(note[1]) if len(note) > 1 else 0.0,
            'offset_prob': float(note[2]) if len(note) > 2 else 0.0,
            'is_onset': bool(is_onset),
            'is_frame': bool(is_frame),
            'strength': float(max(note[0], velocity))  # Overall strength measure
          })
        
        # Console output
        notestr = notename_color(i, not is_frame)
        console_output.append(notestr)
      
      if frame_notes:
        active_notes.extend(frame_notes)
    
    # Print console output
    for notestr in console_output:
        print(notestr, end='')
    print('|')
    
    # Send to WebSocket if enabled and there are clients
    if FLAGS.web_output and websocket_clients and active_notes:
        # Group notes by unique note name (remove duplicates)
        unique_notes = {}
        for note_info in active_notes:
            key = note_info['note']
            if key not in unique_notes or note_info['velocity'] > unique_notes[key]['velocity']:
                unique_notes[key] = note_info
        
        note_data = {
            'timestamp': time.time(),
            'notes': list(unique_notes.values()),
            'serial': serial,
            'total_notes_detected': len(active_notes),
            'frame_info': {
                'frame_threshold': 0.3,
                'onset_threshold': 0.5
            }
        }
        
        # Schedule the broadcast in the WebSocket event loop
        asyncio.run_coroutine_threadsafe(broadcast_notes(note_data), websocket_loop)


def start_websocket_server(loop):
    """Start the WebSocket server in its own thread."""
    asyncio.set_event_loop(loop)
    
    async def run_server():
        print(f"Starting WebSocket server on ws://localhost:{FLAGS.websocket_port}")
        async with websockets.serve(websocket_handler, "localhost", FLAGS.websocket_port):
            # Keep the server running
            await asyncio.Future()  # run forever
    
    try:
        loop.run_until_complete(run_server())
    except KeyboardInterrupt:
        pass


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  # Start WebSocket server if web output is enabled
  websocket_loop = None
  if FLAGS.web_output:
    websocket_loop = asyncio.new_event_loop()
    websocket_thread = threading.Thread(
        target=start_websocket_server, 
        args=(websocket_loop,), 
        daemon=True
    )
    websocket_thread.start()
    print(f"WebSocket server started. Open http://localhost:8000 in your browser")
    print("Make sure to run 'python web_server.py' in another terminal for the HTTP server")

  results = multiprocessing.Queue()
  results_thread = threading.Thread(
      target=result_collector, 
      args=(results, websocket_loop)
  )
  results_thread.start()

  model = tflite_model.Model(model_path=FLAGS.model_path)
  overlap_timesteps = 4
  overlap_wav = model.get_hop_size(
  ) * overlap_timesteps + model.get_window_length()

  if FLAGS.wav_file:
    wav_data = open(FLAGS.wav_file, 'rb').read()
    samples = audio_recorder.wav_data_to_samples(wav_data,
                                                 model.get_sample_rate())
    samples = samples[:model.get_sample_rate() *
                      10]  # Only the first 10 seconds
    samples = samples.reshape((-1, 1))
    samples_length = samples.shape[0]
    # Extend samples with zeros
    samples = np.pad(
        samples, (0, model.get_input_wav_length()), mode='constant')
    for i, pos in enumerate(
        range(0, samples_length - model.get_input_wav_length() + overlap_wav,
              model.get_input_wav_length() - overlap_wav)):
      chunk = samples[pos:pos + model.get_input_wav_length()]
      task = OnsetsTask(AudioChunk(i, chunk))
      task(model)
      results.put(task)
  else:
    tasks = multiprocessing.JoinableQueue()

    ## Make and start the workers
    num_workers = 4
    workers = [
        TfLiteWorker(FLAGS.model_path, tasks, results)
        for i in range(num_workers)
    ]
    for w in workers:
      w.start()

    audio_feeder = AudioQueue(
        callback=lambda audio_chunk: tasks.put(OnsetsTask(audio_chunk)),
        audio_device_index=FLAGS.mic if FLAGS.mic is None else int(FLAGS.mic),
        sample_rate_hz=int(FLAGS.sample_rate_hz),
        model_sample_rate=model.get_sample_rate(),
        frame_length=model.get_input_wav_length(),
        overlap=overlap_wav)

    audio_feeder.start()


def console_entry_point():
  app.run(main)


if __name__ == '__main__':
  console_entry_point()