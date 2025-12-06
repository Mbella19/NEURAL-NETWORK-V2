'use client';

import { useWebSocket } from '@/hooks/useWebSocket';
import { Sidebar } from './Sidebar';

interface ClientLayoutProps {
  children: React.ReactNode;
}

export function ClientLayout({ children }: ClientLayoutProps) {
  // Initialize WebSocket connection
  useWebSocket();

  return (
    <div className="flex h-screen overflow-hidden bg-[#0a0a0b]">
      <Sidebar />
      <main className="flex-1 overflow-hidden">
        {children}
      </main>
    </div>
  );
}
