
<p class="note">Campos con <span class="required">*</span> son necesarios.</p>


    
<div class="forms50cb">    
    <?php echo $form->errorSummary($model); ?>
	<div class="row">
		<?php echo $form->labelEx($model,'TAG'); ?>
<?php echo $form->textField($model,'TAG',array('size'=>50,'maxlength'=>50,)); ?> 
<?php echo $form->error($model,'TAG'); ?>
	</div>

	<div class="row">
		<?php echo $form->labelEx($model,'Tablero'); ?>
<?php echo $form->textField($model,'Tablero',array('size'=>50,'maxlength'=>50,)); ?> 
<?php echo $form->error($model,'Tablero'); ?>
	</div>

<div class="row">

	</div>
    </div>



