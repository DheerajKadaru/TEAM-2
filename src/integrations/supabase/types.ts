export type Json =
  | string
  | number
  | boolean
  | null
  | { [key: string]: Json | undefined }
  | Json[]

export type Database = {
  // Allows to automatically instantiate createClient with right options
  // instead of createClient<Database, { PostgrestVersion: 'XX' }>(URL, KEY)
  __InternalSupabase: {
    PostgrestVersion: "13.0.5"
  }
  public: {
    Tables: {
      alerts: {
        Row: {
          alert_type: string
          created_at: string
          description: string | null
          id: string
          location: Json | null
          metadata: Json | null
          severity: string | null
          species_name: string | null
          status: string | null
          title: string
          updated_at: string
          upload_id: string | null
        }
        Insert: {
          alert_type: string
          created_at?: string
          description?: string | null
          id?: string
          location?: Json | null
          metadata?: Json | null
          severity?: string | null
          species_name?: string | null
          status?: string | null
          title: string
          updated_at?: string
          upload_id?: string | null
        }
        Update: {
          alert_type?: string
          created_at?: string
          description?: string | null
          id?: string
          location?: Json | null
          metadata?: Json | null
          severity?: string | null
          species_name?: string | null
          status?: string | null
          title?: string
          updated_at?: string
          upload_id?: string | null
        }
        Relationships: [
          {
            foreignKeyName: "alerts_upload_id_fkey"
            columns: ["upload_id"]
            isOneToOne: false
            referencedRelation: "uploads"
            referencedColumns: ["id"]
          },
        ]
      }
      analysis_results: {
        Row: {
          abundance_data: Json | null
          asvs: Json | null
          biodiversity_indices: Json | null
          created_at: string
          id: string
          novelty_scores: Json | null
          provenance: Json | null
          summary: string | null
          taxa: Json | null
          upload_id: string
        }
        Insert: {
          abundance_data?: Json | null
          asvs?: Json | null
          biodiversity_indices?: Json | null
          created_at?: string
          id?: string
          novelty_scores?: Json | null
          provenance?: Json | null
          summary?: string | null
          taxa?: Json | null
          upload_id: string
        }
        Update: {
          abundance_data?: Json | null
          asvs?: Json | null
          biodiversity_indices?: Json | null
          created_at?: string
          id?: string
          novelty_scores?: Json | null
          provenance?: Json | null
          summary?: string | null
          taxa?: Json | null
          upload_id?: string
        }
        Relationships: [
          {
            foreignKeyName: "analysis_results_upload_id_fkey"
            columns: ["upload_id"]
            isOneToOne: false
            referencedRelation: "uploads"
            referencedColumns: ["id"]
          },
        ]
      }
      chat_messages: {
        Row: {
          content: string
          created_at: string
          id: string
          metadata: Json | null
          role: string
          session_id: string
        }
        Insert: {
          content: string
          created_at?: string
          id?: string
          metadata?: Json | null
          role: string
          session_id: string
        }
        Update: {
          content?: string
          created_at?: string
          id?: string
          metadata?: Json | null
          role?: string
          session_id?: string
        }
        Relationships: []
      }
      digital_twin_entities: {
        Row: {
          created_at: string
          current_position: Json | null
          entity_type: string
          id: string
          last_updated: string
          metadata: Json | null
          name: string
          status: string | null
          trajectory: Json | null
        }
        Insert: {
          created_at?: string
          current_position?: Json | null
          entity_type: string
          id?: string
          last_updated?: string
          metadata?: Json | null
          name: string
          status?: string | null
          trajectory?: Json | null
        }
        Update: {
          created_at?: string
          current_position?: Json | null
          entity_type?: string
          id?: string
          last_updated?: string
          metadata?: Json | null
          name?: string
          status?: string | null
          trajectory?: Json | null
        }
        Relationships: []
      }
      fisheries: {
        Row: {
          catch_date: string | null
          created_at: string
          id: string
          location: Json | null
          metadata: Json | null
          quantity_kg: number | null
          species_caught: Json | null
          sustainable_rating: string | null
          vessel_name: string
        }
        Insert: {
          catch_date?: string | null
          created_at?: string
          id?: string
          location?: Json | null
          metadata?: Json | null
          quantity_kg?: number | null
          species_caught?: Json | null
          sustainable_rating?: string | null
          vessel_name: string
        }
        Update: {
          catch_date?: string | null
          created_at?: string
          id?: string
          location?: Json | null
          metadata?: Json | null
          quantity_kg?: number | null
          species_caught?: Json | null
          sustainable_rating?: string | null
          vessel_name?: string
        }
        Relationships: []
      }
      reports: {
        Row: {
          created_at: string
          file_path: string | null
          format: string | null
          id: string
          metadata: Json | null
          report_type: string | null
          title: string
          upload_id: string | null
        }
        Insert: {
          created_at?: string
          file_path?: string | null
          format?: string | null
          id?: string
          metadata?: Json | null
          report_type?: string | null
          title: string
          upload_id?: string | null
        }
        Update: {
          created_at?: string
          file_path?: string | null
          format?: string | null
          id?: string
          metadata?: Json | null
          report_type?: string | null
          title?: string
          upload_id?: string | null
        }
        Relationships: [
          {
            foreignKeyName: "reports_upload_id_fkey"
            columns: ["upload_id"]
            isOneToOne: false
            referencedRelation: "uploads"
            referencedColumns: ["id"]
          },
        ]
      }
      stations: {
        Row: {
          created_at: string
          description: string | null
          id: string
          last_sample_date: string | null
          latitude: number
          longitude: number
          metadata: Json | null
          name: string
          station_type: string | null
          status: string | null
        }
        Insert: {
          created_at?: string
          description?: string | null
          id?: string
          last_sample_date?: string | null
          latitude: number
          longitude: number
          metadata?: Json | null
          name: string
          station_type?: string | null
          status?: string | null
        }
        Update: {
          created_at?: string
          description?: string | null
          id?: string
          last_sample_date?: string | null
          latitude?: number
          longitude?: number
          metadata?: Json | null
          name?: string
          station_type?: string | null
          status?: string | null
        }
        Relationships: []
      }
      uploads: {
        Row: {
          created_at: string
          file_path: string
          file_size: number | null
          file_type: string | null
          filename: string
          id: string
          metadata: Json | null
          status: string | null
          updated_at: string
        }
        Insert: {
          created_at?: string
          file_path: string
          file_size?: number | null
          file_type?: string | null
          filename: string
          id?: string
          metadata?: Json | null
          status?: string | null
          updated_at?: string
        }
        Update: {
          created_at?: string
          file_path?: string
          file_size?: number | null
          file_type?: string | null
          filename?: string
          id?: string
          metadata?: Json | null
          status?: string | null
          updated_at?: string
        }
        Relationships: []
      }
    }
    Views: {
      [_ in never]: never
    }
    Functions: {
      [_ in never]: never
    }
    Enums: {
      [_ in never]: never
    }
    CompositeTypes: {
      [_ in never]: never
    }
  }
}

type DatabaseWithoutInternals = Omit<Database, "__InternalSupabase">

type DefaultSchema = DatabaseWithoutInternals[Extract<keyof Database, "public">]

export type Tables<
  DefaultSchemaTableNameOrOptions extends
    | keyof (DefaultSchema["Tables"] & DefaultSchema["Views"])
    | { schema: keyof DatabaseWithoutInternals },
  TableName extends DefaultSchemaTableNameOrOptions extends {
    schema: keyof DatabaseWithoutInternals
  }
    ? keyof (DatabaseWithoutInternals[DefaultSchemaTableNameOrOptions["schema"]]["Tables"] &
        DatabaseWithoutInternals[DefaultSchemaTableNameOrOptions["schema"]]["Views"])
    : never = never,
> = DefaultSchemaTableNameOrOptions extends {
  schema: keyof DatabaseWithoutInternals
}
  ? (DatabaseWithoutInternals[DefaultSchemaTableNameOrOptions["schema"]]["Tables"] &
      DatabaseWithoutInternals[DefaultSchemaTableNameOrOptions["schema"]]["Views"])[TableName] extends {
      Row: infer R
    }
    ? R
    : never
  : DefaultSchemaTableNameOrOptions extends keyof (DefaultSchema["Tables"] &
        DefaultSchema["Views"])
    ? (DefaultSchema["Tables"] &
        DefaultSchema["Views"])[DefaultSchemaTableNameOrOptions] extends {
        Row: infer R
      }
      ? R
      : never
    : never

export type TablesInsert<
  DefaultSchemaTableNameOrOptions extends
    | keyof DefaultSchema["Tables"]
    | { schema: keyof DatabaseWithoutInternals },
  TableName extends DefaultSchemaTableNameOrOptions extends {
    schema: keyof DatabaseWithoutInternals
  }
    ? keyof DatabaseWithoutInternals[DefaultSchemaTableNameOrOptions["schema"]]["Tables"]
    : never = never,
> = DefaultSchemaTableNameOrOptions extends {
  schema: keyof DatabaseWithoutInternals
}
  ? DatabaseWithoutInternals[DefaultSchemaTableNameOrOptions["schema"]]["Tables"][TableName] extends {
      Insert: infer I
    }
    ? I
    : never
  : DefaultSchemaTableNameOrOptions extends keyof DefaultSchema["Tables"]
    ? DefaultSchema["Tables"][DefaultSchemaTableNameOrOptions] extends {
        Insert: infer I
      }
      ? I
      : never
    : never

export type TablesUpdate<
  DefaultSchemaTableNameOrOptions extends
    | keyof DefaultSchema["Tables"]
    | { schema: keyof DatabaseWithoutInternals },
  TableName extends DefaultSchemaTableNameOrOptions extends {
    schema: keyof DatabaseWithoutInternals
  }
    ? keyof DatabaseWithoutInternals[DefaultSchemaTableNameOrOptions["schema"]]["Tables"]
    : never = never,
> = DefaultSchemaTableNameOrOptions extends {
  schema: keyof DatabaseWithoutInternals
}
  ? DatabaseWithoutInternals[DefaultSchemaTableNameOrOptions["schema"]]["Tables"][TableName] extends {
      Update: infer U
    }
    ? U
    : never
  : DefaultSchemaTableNameOrOptions extends keyof DefaultSchema["Tables"]
    ? DefaultSchema["Tables"][DefaultSchemaTableNameOrOptions] extends {
        Update: infer U
      }
      ? U
      : never
    : never

export type Enums<
  DefaultSchemaEnumNameOrOptions extends
    | keyof DefaultSchema["Enums"]
    | { schema: keyof DatabaseWithoutInternals },
  EnumName extends DefaultSchemaEnumNameOrOptions extends {
    schema: keyof DatabaseWithoutInternals
  }
    ? keyof DatabaseWithoutInternals[DefaultSchemaEnumNameOrOptions["schema"]]["Enums"]
    : never = never,
> = DefaultSchemaEnumNameOrOptions extends {
  schema: keyof DatabaseWithoutInternals
}
  ? DatabaseWithoutInternals[DefaultSchemaEnumNameOrOptions["schema"]]["Enums"][EnumName]
  : DefaultSchemaEnumNameOrOptions extends keyof DefaultSchema["Enums"]
    ? DefaultSchema["Enums"][DefaultSchemaEnumNameOrOptions]
    : never

export type CompositeTypes<
  PublicCompositeTypeNameOrOptions extends
    | keyof DefaultSchema["CompositeTypes"]
    | { schema: keyof DatabaseWithoutInternals },
  CompositeTypeName extends PublicCompositeTypeNameOrOptions extends {
    schema: keyof DatabaseWithoutInternals
  }
    ? keyof DatabaseWithoutInternals[PublicCompositeTypeNameOrOptions["schema"]]["CompositeTypes"]
    : never = never,
> = PublicCompositeTypeNameOrOptions extends {
  schema: keyof DatabaseWithoutInternals
}
  ? DatabaseWithoutInternals[PublicCompositeTypeNameOrOptions["schema"]]["CompositeTypes"][CompositeTypeName]
  : PublicCompositeTypeNameOrOptions extends keyof DefaultSchema["CompositeTypes"]
    ? DefaultSchema["CompositeTypes"][PublicCompositeTypeNameOrOptions]
    : never

export const Constants = {
  public: {
    Enums: {},
  },
} as const
