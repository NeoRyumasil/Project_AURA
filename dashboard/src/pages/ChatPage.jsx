import { useState, useEffect, useRef, useCallback } from 'react'
import { supabase } from '../lib/supabaseClient'
import Sidebar from '../components/Sidebar'
import ChatHeader from '../components/ChatHeader'
import ChatFeed from '../components/ChatFeed'
import ChatInput from '../components/ChatInput'
import CallOverlay from '../components/CallOverlay'

const AI_SERVICE = `http://${window.location.hostname}:8000/api/v1`

export default function ChatPage() {
    const [conversations, setConversations] = useState([])
    const [activeConvoId, setActiveConvoId] = useState(null)
    const [messages, setMessages] = useState([])
    const [isCallActive, setIsCallActive] = useState(false)
    const [isSending, setIsSending] = useState(false)
    const feedRef = useRef(null)

    // ─── Load conversations on mount ────────────────
    useEffect(() => {
        loadConversations()
    }, [])

    // ─── Load messages when active conversation changes ──
    useEffect(() => {
        if (activeConvoId) loadMessages(activeConvoId)
        else setMessages([])
    }, [activeConvoId])

    // ─── Auto-scroll on new messages ────────────────
    useEffect(() => {
        if (feedRef.current) {
            feedRef.current.scrollTo({ top: feedRef.current.scrollHeight, behavior: 'smooth' })
        }
    }, [messages])

    // ─── Data fetching ──────────────────────────────
    const loadConversations = async () => {
        const { data } = await supabase
            .from('conversations')
            .select('*')
            .order('updated_at', { ascending: false })
        if (data) setConversations(data)
    }

    const loadMessages = async (convoId) => {
        const { data } = await supabase
            .from('messages')
            .select('*')
            .eq('conversation_id', convoId)
            .order('created_at', { ascending: true })
        if (data) setMessages(data)
    }

    // ─── New chat ───────────────────────────────────
    const handleNewChat = async () => {
        const { data } = await supabase
            .from('conversations')
            .insert({ title: 'New Chat' })
            .select()
            .single()
        if (data) {
            setConversations((prev) => [data, ...prev])
            setActiveConvoId(data.id)
        }
    }

    // ─── Send message ──────────────────────────────
    const handleSend = useCallback(async (text) => {
        if (isSending) return

        // Create conversation if none exists
        let convoId = activeConvoId
        if (!convoId) {
            const { data } = await supabase
                .from('conversations')
                .insert({ title: text.slice(0, 50) })
                .select()
                .single()
            if (!data) return
            convoId = data.id
            setActiveConvoId(convoId)
            setConversations((prev) => [data, ...prev])
        }

        // Optimistically show user message in UI (backend saves to DB)
        const tempUserMsg = { id: `temp-${Date.now()}`, role: 'user', content: text, conversation_id: convoId }
        setMessages((prev) => [...prev, tempUserMsg])

        // Call ai-service — backend handles ALL DB saves
        setIsSending(true)
        try {
            const res = await fetch(`${AI_SERVICE}/chat`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ message: text, conversation_id: convoId }),
            })
            const data = await res.json()
            console.log('[AURA] AI Response:', data)

            // Show AI response in UI (backend already saved to DB)
            const tempAiMsg = {
                id: `temp-ai-${Date.now()}`,
                role: 'aura',
                content: data.text || 'Hmm, the words escaped me~',
                emotion: data.emotion || 'neutral',
                conversation_id: convoId,
                tools_used: data.tools_used || null,
            }
            setMessages((prev) => [...prev, tempAiMsg])

            // Update conversation title on first message
            if (messages.length === 0) {
                await supabase
                    .from('conversations')
                    .update({ title: text.slice(0, 50), updated_at: new Date().toISOString() })
                    .eq('id', convoId)
                loadConversations()
            }
        } catch (err) {
            console.error('[AURA] Chat error:', err)
            setMessages((prev) => [...prev, {
                id: `temp-err-${Date.now()}`,
                role: 'aura',
                content: "Hmm, my connection to the other side seems shaky right now~ Try again?",
                emotion: 'dizzy',
            }])
        } finally {
            setIsSending(false)
        }
    }, [activeConvoId, isSending, messages])

    return (
        <div className="flex h-screen overflow-hidden bg-bg-light text-slate-900">
            <Sidebar
                conversations={conversations}
                activeId={activeConvoId}
                onSelect={setActiveConvoId}
                onNewChat={handleNewChat}
            />

            <main className="flex-1 flex flex-col relative bg-bg-light">
                <ChatHeader onCallStart={() => setIsCallActive(true)} />

                <div ref={feedRef} className="flex-1 overflow-y-auto px-8 py-10 custom-scrollbar">
                    <ChatFeed messages={messages} />
                </div>

                <ChatInput onSend={handleSend} disabled={isSending} />
            </main>

            {isCallActive && <CallOverlay onClose={() => setIsCallActive(false)} />}
        </div>
    )
}