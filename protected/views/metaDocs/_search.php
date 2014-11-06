<div class="wide form">
    <?php
    $this->widget('ext.EChosen.EChosen');
    ?>    

    <?php
    $form = $this->beginWidget('CActiveForm', array(
        'action' => Yii::app()->createUrl($this->route),
        'method' => 'get',
            ));
    $model=new MetaDocs();
    ?>
    <?php
    Yii::app()->clientScript->registerScript('highlightAC', '$.ui.autocomplete.prototype._renderItem = function (ul, item) {
  item.label = item.label.replace(new RegExp("(?![^&;]+;)(?!<[^<>]*)(" + $.ui.autocomplete.escapeRegex(this.term) + ")(?![^<>]*>)(?![^&;]+;)", "gi"), "<strong>$1</strong>");
  return $("<li></li>")
  .data("item.autocomplete", item)
  .append("<a>" + item.label + "</a>")
  .appendTo(ul);
  };', CClientScript::POS_END);
    ?>
    
    <fieldset style="border:1px;border-style:solid;">
        <legend>Oprima Enter o el botón Buscar para iniciar la búsqueda.</legend>
        <table style="padding:0px;margin:0px;width:690px;">
            <tr>
        <td style="margin:0px;width:200px;">
            <?php //echo $form->label($model,'titulo');  ?>
            <?php //echo $form->textField($model,'titulo',array('size'=>60,'maxlength'=>256)); ?>

            <?php echo $form->label($model, 'titulo'); ?>

            <?php
// lee valores de get desde
            if (isset($_GET['titulo'])) {
                $model->titulo = $_GET['titulo'];
            }
            $this->widget('zii.widgets.jui.CJuiAutoComplete', array(
                'name' => 'MetaDocs[titulo]',
                'source' => CController::createUrl('metaDocs/tituloSearch'),
                'options' => array(
                    'minLength' => '1',
                    'select' => 'js:function(event, ui) { console.log(ui.item.id +":"+ui.item.value); }',
                ),
                'htmlOptions' => array(
                    'style' => 'height:17px;width:200px;'
                ),
                'model' => $model,
                'value' => $model->titulo,
            ));
            ?>
        </td>
        <td style="padding:0px;margin:0px;width:200px;">
            <?php echo $form->label($model, 'autores'); ?>
            <?php
// lee valores de get desde
            if (isset($_GET['autores'])) {
                $model->titulo = $_GET['autores'];
            }
            $this->widget('zii.widgets.jui.CJuiAutoComplete', array(
                'name' => 'MetaDocs[autores]',
                'source' => CController::createUrl('metaDocs/autoresSearch'),
                'options' => array(
                    'minLength' => '1',
                    'select' => 'js:function(event, ui) { console.log(ui.item.id +":"+ui.item.value); }',
                ),
                'htmlOptions' => array(
                    'style' => 'height:17px;width:200px;'
                ),
                'model' => $model,
                'value' => $model->autores,
            ));
            ?>
        </td>


        <td style="padding:0px;margin:0px;width:200px;">
            <?php echo $form->label($model, 'fabricante'); ?>
            <?php
            $datos = array();
            $modelos = Fabricantes::model()->findAllBySql("select id,descripcion from fabricantes");
            foreach ($modelos as $modelo) {
                $datos = $datos + array($modelo->id => $modelo->descripcion);
            }
            ?>
<?php echo $form->dropDownList($model, 'fabricante', $datos, array('empty'=>'Escriba/Seleccione','empty'=>'Escriba/Seleccione','class' => 'chzn-select', 'style' => 'width:200px;', 'maxlength' => 64)); ?>
        </td>
        </tr>
                <tr>

        <td >
            <?php echo $form->label($model, 'descripcion'); ?>
                        <?php
// lee valores de get desde
            if (isset($_GET['descripcion'])) {
                $model->titulo = $_GET['descripcion'];
            }
            $this->widget('zii.widgets.jui.CJuiAutoComplete', array(
                'name' => 'MetaDocs[descripcion]',
                'source' => CController::createUrl('metaDocs/descripcionSearch'),
                'options' => array(
                    'minLength' => '1',
                    'select' => 'js:function(event, ui) { console.log(ui.item.id +":"+ui.item.value); }',
                ),
                'htmlOptions' => array(
                    'style' => 'height:17px;width:200px;'
                ),
                'model' => $model,
                'value' => $model->descripcion,
            ));
            ?>
        </td>


        <td >
            <?php echo $form->label($model, 'medio'); ?>
            <?php
            $datos = array();
            $modelos = Medios::model()->findAllBySql("select id,descripcion from medios");
            foreach ($modelos as $modelo) {
                $datos = $datos + array($modelo->id => $modelo->descripcion);
            }
            ?>
            <?php echo $form->dropDownList($model, 'medio', $datos, array('empty'=>'Escriba/Seleccione','class' => 'chzn-select', 'style' => 'width:200px;', 'maxlength' => 64)); ?>

        </td>

        <td >
            <?php echo $form->label($model, 'idioma'); ?>
             <?php
            $datos = array();
            $modelos = Idiomas::model()->findAllBySql("select id,descripcion from idiomas");
            foreach ($modelos as $modelo) {
                $datos = $datos + array($modelo->id => $modelo->descripcion);
            }
            ?>
            <?php echo $form->dropDownList($model, 'idioma', $datos, array('empty'=>'Escriba/Seleccione','class' => 'chzn-select', 'style' => 'width:200px;', 'maxlength' => 64)); ?>

        </td>
        </tr>

        <tr>

        

        <td >
            
            
            <?php echo $form->label($model, 'numPedido'); ?>
            <?php
// lee valores de get desde
            if (isset($_GET['numPedido'])) {
                $model->titulo = $_GET['numPedido'];
            }
            $this->widget('zii.widgets.jui.CJuiAutoComplete', array(
                'name' => 'MetaDocs[numPedido]',
                'source' => CController::createUrl('metaDocs/numPedidoSearch'),
                'options' => array(
                    'minLength' => '1',
                    'select' => 'js:function(event, ui) { console.log(ui.item.id +":"+ui.item.value); }',
                ),
                'htmlOptions' => array(
                    'style' => 'height:17px;width:200px;'
                ),
                'model' => $model,
                'value' => $model->numPedido,
            ));
            ?>       </td>

        <td >
            <?php echo $form->label($model, 'numComision'); ?>
            <?php
// lee valores de get desde
            if (isset($_GET['numComision'])) {
                $model->titulo = $_GET['numComision'];
            }
            $this->widget('zii.widgets.jui.CJuiAutoComplete', array(
                'name' => 'MetaDocs[numComision]',
                'source' => CController::createUrl('metaDocs/numComisionSearch'),
                'options' => array(
                    'minLength' => '1',
                    'select' => 'js:function(event, ui) { console.log(ui.item.id +":"+ui.item.value); }',
                ),
                'htmlOptions' => array(
                    'style' => 'height:17px;width:200px;'
                ),
                'model' => $model,
                'value' => $model->numComision,
            ));
            ?>        </td>

        </tr>
        <tr>
        <td >
            <?php echo $form->label($model, 'ubicacionT'); ?>
            <?php
            $datos = array();
            $modelos = UbicacionTec::model()->findAllBySql("select id,descripcion from ubicacionTec");
            foreach ($modelos as $modelo) {
                $datos = $datos + array($modelo->id => $modelo->descripcion);
            }
            ?>
            <?php echo $form->dropDownList($model, 'ubicacionT', $datos, array('empty'=>'Escriba/Seleccione','class' => 'chzn-select', 'style' => 'width:200px;', 'maxlength' => 64)); ?>
        </td>

        <td >
            <?php echo $form->label($model, 'cerveceria'); ?>
            <?php
            $datos = array();
            $modelos = Cervecerias::model()->findAllBySql("select id,descripcion from cervecerias");
            foreach ($modelos as $modelo) {
                $datos = $datos + array($modelo->id => $modelo->descripcion);
            }
            ?>
            <?php echo $form->dropDownList($model, 'cerveceria', $datos, array('empty'=>'Escriba/Seleccione','class' => 'chzn-select', 'style' => 'width:200px;', 'maxlength' => 64)); ?>
        </td>


        <td >
            <?php echo $form->label($model, 'tipoContenido'); ?>
             <?php
            $datos = array();
            $modelos = TipoContenidos::model()->findAllBySql("select id,descripcion from tipoContenidos");
            foreach ($modelos as $modelo) {
                $datos = $datos + array($modelo->id => $modelo->descripcion);
            }
            ?>
            <?php echo $form->dropDownList($model, 'tipoContenido', $datos, array('empty'=>'Escriba/Seleccione','class' => 'chzn-select', 'style' => 'width:200px;', 'maxlength' => 64)); ?>
        </td>
        </tr>
        <tr>
        <td >
            <?php echo $form->label($model, 'documento'); ?>
            <?php
            $datos = array();
            $modelos = Documentos::model()->findAllBySql("select id,descripcion from documentos");
            foreach ($modelos as $modelo) {
                $datos = $datos + array($modelo->id => $modelo->descripcion);
            }
            ?>
            <?php echo $form->dropDownList($model, 'documento', $datos, array('empty'=>'Escriba/Seleccione','class' => 'chzn-select', 'style' => 'width:200px;', 'maxlength' => 64)); ?>

        </td>
                <td  style="padding:0px;margin:0px;width:100px;">
            <?php echo $form->label($model, 'ISBN'); ?>
            <?php echo $form->textField($model, 'ISBN', array('style' => 'width:200px;padding:0px;margin:0px;', 'maxlength' => 32)); ?>
        </td>

        <td  style="padding:0px;margin:0px;width:100px;">
            <?php echo $form->label($model, 'EAN13'); ?>
            <?php echo $form->textField($model, 'EAN13', array('style' => 'width:200px;padding:0px;margin:0px;', 'maxlength' => 32)); ?>
        </td>
</tr>
</table>            
        

    </fieldset>
                <hr/>
            <?php echo CHtml::submitButton(Yii::t('app', 'Buscar')); ?>
    

        <?php $this->endWidget(); ?>
</div><!-- search-form -->
