package costream.plan.executor.application.smartgrid;

import org.apache.storm.tuple.Values;
import org.apache.storm.utils.Utils;

public class StreamValues extends Values {
    private Object messageId;
    private String streamId = Utils.DEFAULT_STREAM_ID;

    public StreamValues() {
    }

    public StreamValues(Object... vals) {
        super(vals);
    }

    public Object getMessageId() {
        return messageId;
    }

    public void setMessageId(Object messageId) {
        this.messageId = messageId;
    }

    public String getStreamId() {
        return streamId;
    }

    public void setStreamId(String streamId) {
        this.streamId = streamId;
    }

}
