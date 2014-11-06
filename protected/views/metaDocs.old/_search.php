<div class="wide form">

<?php $form=$this->beginWidget('CActiveForm', array(
        'action'=>Yii::app()->createUrl($this->route),
        'method'=>'get',
)); ?>

        <div class="row">
                <?php echo $form->label($model,'id'); ?>
                <?php echo $form->textField($model,'id',array('size'=>20,'maxlength'=>20)); ?>
        </div>

        <div class="row">
                <?php echo $form->label($model,'tipoContenido'); ?>
                <?php ; ?>
        </div>

        <div class="row">
                <?php echo $form->label($model,'fabricante'); ?>
                <?php ; ?>
        </div>

        <div class="row">
                <?php echo $form->label($model,'cerveceria'); ?>
                <?php ; ?>
        </div>

        <div class="row">
                <?php echo $form->label($model,'numPedido'); ?>
                <?php echo $form->textField($model,'numPedido',array('size'=>60,'maxlength'=>64)); ?>
        </div>

        <div class="row">
                <?php echo $form->label($model,'numComision'); ?>
                <?php echo $form->textField($model,'numComision',array('size'=>60,'maxlength'=>64)); ?>
        </div>

        <div class="row">
                <?php echo $form->label($model,'ubicacionT'); ?>
                <?php ; ?>
        </div>

        <div class="row">
                <?php echo $form->label($model,'descripcion'); ?>
                <?php echo $form->textField($model,'descripcion',array('size'=>60,'maxlength'=>256)); ?>
        </div>

        <div class="row">
                <?php echo $form->label($model,'titulo'); ?>
                <?php echo $form->textField($model,'titulo',array('size'=>60,'maxlength'=>256)); ?>
        </div>

        <div class="row">
                <?php echo $form->label($model,'version'); ?>
                <?php echo $form->textField($model,'version',array('size'=>60,'maxlength'=>64)); ?>
        </div>

        <div class="row">
                <?php echo $form->label($model,'medio'); ?>
                <?php ; ?>
        </div>

        <div class="row">
                <?php echo $form->label($model,'idioma'); ?>
                <?php ; ?>
        </div>

        <div class="row">
                <?php echo $form->label($model,'disponibles'); ?>
                <?php echo $form->textField($model,'disponibles'); ?>
        </div>

        <div class="row">
                <?php echo $form->label($model,'existencias'); ?>
                <?php echo $form->textField($model,'existencias'); ?>
        </div>

        <div class="row">
                <?php echo $form->label($model,'modulo'); ?>
                <?php echo $form->textField($model,'modulo'); ?>
        </div>

        <div class="row">
                <?php echo $form->label($model,'columna'); ?>
                <?php echo $form->textField($model,'columna'); ?>
        </div>

        <div class="row">
                <?php echo $form->label($model,'fila'); ?>
                <?php echo $form->textField($model,'fila'); ?>
        </div>

        <div class="row">
                <?php echo $form->label($model,'documento'); ?>
                <?php ; ?>
        </div>

        <div class="row">
                <?php echo $form->label($model,'ruta'); ?>
                <?php echo $form->textField($model,'ruta',array('size'=>60,'maxlength'=>256)); ?>
        </div>

        <div class="row">
                <?php echo $form->label($model,'fechaCreacion'); ?>
                <?php echo $form->textField($model,'fechaCreacion'); ?>
        </div>

        <div class="row">
                <?php echo $form->label($model,'fechaRecepcion'); ?>
                <?php echo $form->textField($model,'fechaRecepcion'); ?>
        </div>

        <div class="row">
                <?php echo $form->label($model,'autores'); ?>
                <?php echo $form->textField($model,'autores',array('size'=>60,'maxlength'=>256)); ?>
        </div>

        <div class="row">
                <?php echo $form->label($model,'usuario'); ?>
                <?php ; ?>
        </div>

        <div class="row">
                <?php echo $form->label($model,'revisado'); ?>
                <?php echo $form->checkBox($model,'revisado'); ?>
        </div>

        <div class="row">
                <?php echo $form->label($model,'userRevisado'); ?>
                <?php ; ?>
        </div>

        <div class="row">
                <?php echo $form->label($model,'fechaRevisado'); ?>
                <?php echo $form->textField($model,'fechaRevisado'); ?>
        </div>

        <div class="row">
                <?php echo $form->label($model,'eliminado'); ?>
                <?php echo $form->checkBox($model,'eliminado'); ?>
        </div>

        <div class="row">
                <?php echo $form->label($model,'secuencia'); ?>
                <?php ; ?>
        </div>

        <div class="row">
                <?php echo $form->label($model,'ordenSecuencia'); ?>
                <?php ; ?>
        </div>
    
    <div class="row">
                <?php echo $form->label($model,'ISBN'); ?>
                <?php echo $form->textField($model,'ISBN',array('size'=>32,'maxlength'=>32)); ?>
        </div>

        <div class="row">
                <?php echo $form->label($model,'EAN13'); ?>
                <?php echo $form->textField($model,'EAN13',array('size'=>32,'maxlength'=>32)); ?>
        </div>

        <div class="row buttons">
                <?php echo CHtml::submitButton(Yii::t('app', 'Buscar')); ?>
        </div>

<?php $this->endWidget(); ?>

</div><!-- search-form -->
