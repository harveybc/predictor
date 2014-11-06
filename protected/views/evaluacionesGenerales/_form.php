<p class="note">Campos con<span class="required">*</span> son necesarios.</p>

<?php echo $form->errorSummary($model); ?>

<div styler="width:400px!important;height:150px;margin-left:0px;margin-bottom:5px !important;border-color:#961C1F;padding-top:10px;padding-right:5px;padding-left:14px;">

<div>
    <div>
        <div>
            <div class="row">
		<?php echo $form->labelEx($model,'fechaInicio'); ?>
<?php $this->widget('zii.widgets.jui.CJuiDatePicker',
						 array(
								 'model'=>'$model',
								 'name'=>'EvaluacionesGenerales[fechaInicio]',
								 //'language'=>'de',
								 'value'=>$model->fechaInicio,
								 'htmlOptions'=>array('size'=>10, 'style'=>'width:80px !important'),
									 'options'=>array(
									 'showButtonPanel'=>true,
									 'changeYear'=>true,                                      
									 'changeYear'=>true,
									 ),
								 )
							 );
					; ?>
<?php echo $form->error($model,'fechaInicio'); ?>
	</div>

            </div>
            <div>
                <div class="row">
		<?php echo $form->labelEx($model,'fechaFin'); ?>
<?php $this->widget('zii.widgets.jui.CJuiDatePicker',
						 array(
								 'model'=>'$model',
								 'name'=>'EvaluacionesGenerales[fechaFin]',
								 //'language'=>'de',
								 'value'=>$model->fechaFin,
								 'htmlOptions'=>array('size'=>10, 'style'=>'width:80px !important'),
									 'options'=>array(
									 'showButtonPanel'=>true,
									 'changeYear'=>true,                                      
									 'changeYear'=>true,
									 ),
								 )
							 );
					; ?>
<?php echo $form->error($model,'fechaFin'); ?>
	</div>

            </div>
            </div>
            </div>
<div>
    <div>
            <div>
                <div class="row">
		<?php echo $form->labelEx($model,'descripcion'); ?>
<?php echo $form->textField($model,'descripcion',array('size'=>60,'maxlength'=>128)); ?>
<?php echo $form->error($model,'descripcion'); ?>
	</div>
            </div>
        </div>
    </div>

</div>	

	
	
	<div class="row">
		<?php echo $form->labelEx($model,'pregunta1'); ?>
<?php echo $form->textField($model,'pregunta1'); ?>
<?php echo $form->error($model,'pregunta1'); ?>
	</div>

	<div class="row">
		<?php echo $form->labelEx($model,'pregunta2'); ?>
<?php echo $form->textField($model,'pregunta2'); ?>
<?php echo $form->error($model,'pregunta2'); ?>
	</div>

	<div class="row">
		<?php echo $form->labelEx($model,'pregunta3'); ?>
<?php echo $form->textField($model,'pregunta3'); ?>
<?php echo $form->error($model,'pregunta3'); ?>
	</div>

	<div class="row">
		<?php echo $form->labelEx($model,'pregunta4'); ?>
<?php echo $form->textField($model,'pregunta4'); ?>
<?php echo $form->error($model,'pregunta4'); ?>
	</div>

	<div class="row">
		<?php echo $form->labelEx($model,'pregunta5'); ?>
<?php echo $form->textField($model,'pregunta5'); ?>
<?php echo $form->error($model,'pregunta5'); ?>
	</div>

	<div class="row">
		<?php echo $form->labelEx($model,'pregunta6'); ?>
<?php echo $form->textField($model,'pregunta6'); ?>
<?php echo $form->error($model,'pregunta6'); ?>
	</div>

	<div class="row">
		<?php echo $form->labelEx($model,'pregunta7'); ?>
<?php echo $form->textField($model,'pregunta7'); ?>
<?php echo $form->error($model,'pregunta7'); ?>
	</div>

	<div class="row">
		<?php echo $form->labelEx($model,'pregunta8'); ?>
<?php echo $form->textField($model,'pregunta8'); ?>
<?php echo $form->error($model,'pregunta8'); ?>
	</div>

	<div class="row">
		<?php echo $form->labelEx($model,'pregunta9'); ?>
<?php echo $form->textField($model,'pregunta9'); ?>
<?php echo $form->error($model,'pregunta9'); ?>
	</div>

	<div class="row">
		<?php echo $form->labelEx($model,'pregunta10'); ?>
<?php echo $form->textField($model,'pregunta10'); ?>
<?php echo $form->error($model,'pregunta10'); ?>
	</div>


